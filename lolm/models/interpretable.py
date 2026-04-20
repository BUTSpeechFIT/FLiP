from math import sqrt
import numpy as np
import torch
from typing import Tuple
from lolm.utils import move_to_device


class BayLoLM(torch.nn.Module):
    """A Bayesian Log-Linear Model for interpreting multilingual, multimodal
    sentence embeddings using bag-of-words"""

    def __init__(self, W: int, K: int, prior_variance: float = 0.1, R: int = 4):
        """Initialize our model

        Args:
            W (int): vocab size or number of words
            K (int): embedding dimension
            prior_variance (float): Prior variance
            R (int):  Number of Monte-Carlo samples for the reparam-trick
        """

        super().__init__()

        self.W = W  # vocab size
        self.K = K  # embedding dimension

        # we wrap everything in nn.Parameter, and explicitly specify requires_grad=False,
        # because when we call model.cuda(), all of nn.Parameters go to cuda
        self.prior_variance = torch.nn.Parameter(
            torch.FloatTensor([prior_variance]), requires_grad=False
        )
        self.R = R  # Number of Monte-Carlo samples for the re-param trick

        self.K_float = torch.nn.Parameter(
            torch.FloatTensor([self.K]), requires_grad=False
        )
        self.R_float = torch.nn.Parameter(
            torch.FloatTensor([self.R]), requires_grad=False
        )

        # word embeddings matrix / subspace
        # or mean of the variational (posterior) distribution of word embeddings
        self.E = torch.nn.Parameter(torch.zeros(W, K).to(dtype=torch.float32))
        torch.nn.init.normal_(self.E.data, mean=0, std=sqrt(prior_variance))

        # log variance of the variational (posterior) distribution
        self.log_var = torch.nn.Parameter(
            torch.ones(K) * torch.log(self.prior_variance)
        ).to(dtype=torch.float32)

        # bias vector
        self.b = torch.nn.Parameter(torch.randn(W, 1).to(dtype=torch.float32))
        torch.nn.init.xavier_uniform_(self.b)

    def __repr__(self):
        s = f"""Model params
        Word_embs    : {self.E.shape}, Requires grad: {self.E.requires_grad}
        Log_variance : {self.log_var.shape},    Requires grad: {self.log_var.requires_grad}
        bias_vector  : {self.b.shape}, Requires grad: {self.b.requires_grad}
        """
        return s

    def init_bias_with_log_unigram_dist(self, X):
        r"""We will initialize the bias vector with log of unigram distribution over vocabulary.
        This should help us with better initialization and faster convergence.

        b = \log (\sum_d x_d) / (\sum_d \sum_i x_{di})
        """

        # if X is sparse matrix, X.A gives the dense version of it in numpy array format
        # if isinstance(X, np.ndarray):
        #    X = X + 1e-08  # to avoid zeros
        # else:
        #    X = X.A + 1e-08  # to avoid any zeros

        # we would like b to of size (W, 1)
        self.b[:, 0].data = torch.from_numpy(np.log(X.sum(axis=0) / X.sum()))

    def sample(self):
        """Sample from std.normal for re-parametrization trick"""

        eps = torch.randn(self.R, self.W, self.K).to(device=self.R_float.device)
        return eps

    def compute_kld(self):
        r"""Compute KL divergence from variational posterior to the prior.
        The variational posterior for each word embedding has a specific mean, but
        shares the same diagonal variance. The prior is centered around 0, with isotropic
        variance.

        For full cov Gaussian dists
        D_KL( N(\nu, \Gamma^-1) || N(\mu, \Lambda^-1) ) =
        0.5[ tr(\Lambda Gamma.inv()) + \log|\Gamma| - log|\Lambda| +
            (\nu - \mu)^T \Gamma (\nu - \mu) - K ]

        From diagonal cov to isotropic cov Gaussian dists
        D_KL( N(\nu, \diag(\exp(\log_sigma)) ) || N(0, \lambda I) =
        0.5[ (exp(log_sigma).sum() / prior_variance) + (K * log(prior_var)) - (log_var.sum()) +
        ((\nu * \nu).sum() / prior_variance) - K ]
        """

        # we multiply certain terms with self.W because that is our vocab size and we need KLD
        # for each var dist corresponding to a word in vocab
        term_1 = (torch.exp(self.log_var).sum() / self.prior_variance) * self.W
        term_2 = (
            (self.K_float * torch.log(self.prior_variance)) - self.log_var.sum()
        ) * self.W
        term_3 = (self.E**2).sum() / self.prior_variance
        term_4 = -1.0 * self.K_float * self.W
        kld = (term_1 + term_2 + term_3 + term_4) * 0.5
        # print(term_1, term_2, term_3, term_4)
        return kld

    def forward(self, doc_embs: torch.FloatTensor) -> torch.FloatTensor:
        r"""Compute log of thetas, where each theta_d is the unigram distribution over
        document `d` estiamted from the current params (word-embedding matrix, bias vector)
        and given sentence/doc embedding.

        Args:
            doc_embs (torch.FloatTensor): Document or sentence embeddings extracted from a model.
                Shape is n_docs x emb_dim
        """

        eps = self.sample()  # R x W x K
        aux = (eps * torch.exp(self.log_var * 0.5)) + self.E  # R x W x K

        lse_term = self.b + (aux @ doc_embs.T)  # R x W x D
        lse_term = lse_term.sum(dim=0) / self.R_float
        lse = torch.logsumexp(lse_term, dim=0)

        numerator = (self.E @ doc_embs.T) + self.b  # W x D

        log_thetas = numerator - lse  # W x D

        return log_thetas.T  #  D x W

    def compute_exp_log_likelihood(
        self,
        rixs: torch.LongTensor,
        cixs: torch.LongTensor,
        vals: torch.FloatTensor,
        doc_embs: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Compute expected log-likelihood of the data, using the current parameters,
        given the sentence embeddings.

        Args:
            rixs (torch.LongTensor): Row indices (docs)
            cixs (torch.LongTensor): Col indices (words)
            vals (torch.FloatTensor): Counts (occurrences)
            doc_embs (torch.FloatTensor): Document embeddings corresponding to the rows

        Returns:
            float: exp. log-likelihood of the data
        """

        log_thetas = self.forward(doc_embs)

        exp_llh = (log_thetas[rixs, cixs] * vals).sum()

        return exp_llh

    def compute_elbo(
        self,
        rixs: torch.LongTensor,
        cixs: torch.LongTensor,
        vals: torch.FloatTensor,
        doc_embs: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Compute evidence lower-bound which is
        exp_log-likelihood of the data - KL_divergence, using the current parameters,
        given the sentence embeddings.

        Args:
            rixs (torch.LongTensor): Row indices (docs)
            cixs (torch.LongTensor): Col indices (words)
            vals (torch.FloatTensor): Counts (occurrences)
            doc_embs (torch.FloatTensor): Document embeddings corresponding to the rows

        Returns:
            float, float: exp. log-likelihood of the data, KLD
        """

        exp_llh = self.compute_exp_log_likelihood(rixs, cixs, vals, doc_embs)
        kl_div = self.compute_kld()

        elbo = (exp_llh, kl_div)

        return elbo


class LoLM(torch.nn.Module):
    """A Log-Linear Model for interpreting multilingual, multimodal
    sentence embeddings using bag-of-words"""

    def __init__(self, W: int, K: int):
        """Initialize LoL model

        Args:
            W: vocab size or number of words
            K: embedding dimension
        """

        super().__init__()

        self.W = W
        self.K = K

        # word embeddings matrix / subspace
        self.E = torch.nn.Parameter(torch.randn(W, K).to(dtype=torch.float32))
        torch.nn.init.xavier_normal_(self.E.data)

        # bias vector
        self.b = torch.nn.Parameter(torch.randn(W, 1).to(dtype=torch.float32))
        torch.nn.init.xavier_normal_(self.b)

    def __repr__(self):
        s = f"""Model params
        Word embs   (E): {self.E.shape}, Requires grad: {self.E.requires_grad}
        bias vector (b): {self.b.shape}, Requires grad: {self.b.requires_grad}
        """
        return s

    def init_bias_with_log_unigram_dist(self, X):
        r"""We will initialize the bias vector with log of unigram distribution over vocabulary.
        This should help us with better initialization.

        b = \log (\sum_d x_d) / (\sum_d \sum_i x_{di})
        """
        # we would like b to of size (W, 1)
        self.b[:, 0].data = torch.from_numpy(
            np.log((X.sum(axis=0) + 1e-8) / (X.sum() + (1e-8 * self.W)))
        )

    def forward(self, doc_embs):
        r"""Compute log of thetas, where theta_d is the unigram distribution over
        document `d` estimated from the current params (word-embedding matrix, bias vector)
        and given document embedding for `d`.
        """

        mat = self.b + (self.E @ doc_embs.T)  # shape is W x D  vocab_size x n_docs
        mat = mat.T  # shape is D x W
        log_thetas = torch.nn.functional.log_softmax(mat, dim=1)  # shape is D x W

        return log_thetas

    def compute_neg_log_likelihood(self, rixs, cixs, vals, doc_embs):
        """Compute negative log-likelihood of the data, given the current parameters / embeddings

        Args:
            rixs (torch.LongTensor): Row indices (docs)
            cixs (torch.LongTensor): Col indices (words)
            vals (torch.FloatTensor): counts (occurrences)
            doc_embs (torch.FloatTensor): Document embeddings corresponding to the rows

        Returns:
            float: log-likelihood of the data
        """

        log_thetas = self.forward(doc_embs)

        llh = (log_thetas[rixs, cixs] * vals).sum()

        return -1.0 * llh

    def compute_l1_penalty(self) -> torch.FloatTensor:
        """Compute L1 penalty on the word embeddings"""
        return self.E.abs().sum()

    def apply_proximal_operator(self, lambda_reg: float, lr: float) -> None:
        """Apply proximal operator (soft-thresholding) to word embeddings for L1 regularization.

        This implements the proximal gradient method for L1:
        After gradient step on smooth part (neg. log-likelihood), apply:
            E := soft_threshold(E, lr * lambda_reg)

        Soft-thresholding:
            E[i,j] := sign(E[i,j]) * max(|E[i,j]| - lr*lambda_reg, 0)

        This achieves exact sparsity: sets small values to exactly 0.

        Args:
            lambda_reg: L1 regularization coefficient
            lr: Learning rate (step size)

        Note: This should be called AFTER optimizer.step() when using proximal gradient.
              Do NOT include L1 penalty in the loss when using this method.
        """
        threshold = lr * lambda_reg
        with torch.no_grad():
            # Soft-thresholding: sign(x) * max(|x| - threshold, 0)
            self.E.data = torch.sign(self.E.data) * torch.clamp(
                torch.abs(self.E.data) - threshold, min=0.0
            )


class FactLoLM(LoLM):
    """A Factorized Log-Linear Model for interpreting multilingual, multimodal
    sentence embeddings using bag-of-n-grams.

    Inherits init_bias_with_log_unigram_dist and compute_neg_log_likelihood from LoLM.
    """

    def __init__(self, W: int, K: int, rank: int):
        """Initialize LoL model

        Args:
            W (int): vocab size or number of words
            K (int): embedding dimension
            rank (int): for low rank factorization of embedding matrix
        """

        # Skip LoLM.__init__ to avoid creating self.E
        torch.nn.Module.__init__(self)

        self.W = W
        self.K = K
        self.rank = rank

        # word embeddings matrix / subspace
        self.E1 = torch.nn.Parameter(torch.randn(W, rank).to(dtype=torch.float32))
        torch.nn.init.xavier_normal_(self.E1.data)
        self.E2 = torch.nn.Parameter(torch.randn(rank, K).to(dtype=torch.float32))
        torch.nn.init.xavier_normal_(self.E2.data)

        # bias vector
        self.b = torch.nn.Parameter(torch.randn(W, 1).to(dtype=torch.float32))
        torch.nn.init.xavier_normal_(self.b)

    def __repr__(self):
        s = f"""Model params
        Word embs  : {self.E1.shape}, {self.E2.shape} Requires grad: {self.E1.requires_grad}{self.E2.requires_grad}
        bias vector: {self.b.shape}, Requires grad: {self.b.requires_grad}
        """
        return s

    def forward(self, doc_embs):
        r"""Compute log of thetas, where theta_d is the unigram distribution over
        document `d` estimated from the current params (word-embedding matrix, bias vector)
        and given document embedding for `d`.
        """

        mat = self.b + (
            self.E1 @ (self.E2 @ doc_embs.T)
        )  # shape is W x D  vocab_size x n_docs
        mat = mat.T  # shape is D x W
        log_thetas = torch.nn.functional.log_softmax(mat, dim=1)  # shape is D x W

        return log_thetas

    def compute_l1_penalty(self) -> torch.FloatTensor:
        """Compute L1 penalty on the low rank word embeddings E1"""
        return self.E1.abs().sum()

    def apply_proximal_operator(self, lambda_reg: float, lr: float) -> None:
        """Apply proximal operator (soft-thresholding) to low-rank embeddings for L1 regularization.

        This implements the proximal gradient method for L1:
        After gradient step on smooth part (neg. log-likelihood), apply:
            E1 := soft_threshold(E1, lr * lambda_reg)

        Soft-thresholding:
            E1[i,j] := sign(E1[i,j]) * max(|E1[i,j]| - lr*lambda_reg, 0)

        This achieves exact sparsity in the low-rank factor E1.

        Args:
            lambda_reg: L1 regularization coefficient
            lr: Learning rate (step size)

        Note: This should be called AFTER optimizer.step() when using proximal gradient.
              Do NOT include L1 penalty in the loss when using this method.
              We only regularize E1 (the first low-rank factor) for efficiency.
        """
        threshold = lr * lambda_reg
        with torch.no_grad():
            # Soft-thresholding: sign(x) * max(|x| - threshold, 0)
            self.E1.data = torch.sign(self.E1.data) * torch.clamp(
                torch.abs(self.E1.data) - threshold, min=0.0
            )


@torch.no_grad()
def compute_llh_of_data(model, dloader, device):
    """Compute LLH of the entire data"""
    nllh = 0.0
    for batch in dloader:
        batch = move_to_device(batch, device)
        nllh += model.compute_neg_log_likelihood(
            batch["rixs"], batch["cixs"], batch["vals"], batch["doc_embs1"]
        )
    return -1.0 * nllh.item()  # gives the llh


class LayerWiseLoLM(torch.nn.Module):
    """A LoLM model for each layer in a pre-trained LM"""

    def __init__(self, n_layers: int, vocab_size: int, dim: int):
        """Initialise a BoW model for each layer"""

        super().__init__()

        self.n_layers = n_layers

        self.layer2lolm = torch.nn.ModuleList(
            [LoLM(vocab_size, dim) for i in range(n_layers)]
        )

    def forward(self, layer_ix: int, doc_embs: torch.FloatTensor):
        """Forward pass through one layer-specific model to obtain log_thetas
        log of estimated unigram probability distribution"""

        return self.layer2lolm[layer_ix](doc_embs)

    def compute_neg_log_likelihood(self, layer_ix, rixs, cixs, vals, doc_embs):
        """Compute negative log-likelihood of the data, given the current parameters / embeddings"""

        return self.layer2lolm[layer_ix].compute_neg_log_likelihood(
            rixs, cixs, vals, doc_embs
        )


# test methods for the above models


def test_baylolm(args):
    """Test BayLoLM model"""

    vocab_size = 10
    emb_dim = 8
    prior_var = 0.1
    n_samples = 4
    lol = BayLoLM(vocab_size, emb_dim, prior_var, n_samples)

    # print("total params {:f} trainable {:f}".format(*train_utils.get_num_params(lol)))

    device = torch.device("cuda" if args.cuda else "cpu")
    if args.cuda:
        lol.cuda()

    with torch.no_grad():
        kld = lol.compute_kld()
        print("kld where mean is randomly init:", kld)

        lol.E.data.zero_()
        kld = lol.compute_kld()
        print("kld where mean is zero-ed: ", kld)

        eps = lol.sample()
        print("eps:", eps.shape, eps.device)

    print("Toy dataset from 20NG")

    categories = [
        "alt.atheism",
        "talk.religion.misc",
    ]

    data_train = fetch_20newsgroups(
        subset="train",
        categories=categories,
        shuffle=False,
        random_state=args.seed,
        remove=("headers", "footers", "quotes"),
    )

    cvect = CountVectorizer(min_df=2, max_features=vocab_size, stop_words="english")
    dbyw = cvect.fit_transform(data_train.data).tocsr()

    print("DbyW:", dbyw.shape)

    rixs, cixs = dbyw.nonzero()
    rixs = torch.from_numpy(rixs).long().to(device=device)
    cixs = torch.from_numpy(cixs).long().to(device=device)
    vals = torch.from_numpy(dbyw.data).float().to(device=device)

    n_docs = dbyw.shape[0]
    doc_embs = torch.randn(size=(n_docs, emb_dim)).to(device=device)

    exp_llh = lol.compute_exp_log_likelihood(rixs, cixs, vals, doc_embs)
    kld = lol.compute_kld()
    print(exp_llh.item(), kld.item())
    print(lol.compute_elbo(rixs, cixs, vals, doc_embs))


if __name__ == "__main__":
    import argparse
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import CountVectorizer

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test", type=str, default="", choices=["baylolm"], help="which model to test"
    )
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if args.test == "baylolm":
        test_baylolm(args)
