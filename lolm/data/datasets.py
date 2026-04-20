import json
import sys
import os
import logging
from typing import Union, List
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
import scipy.sparse

import torch
import torchaudio
from torch.utils.data import Dataset, Sampler
import torch.multiprocessing
from lolm.data.utils import filter_data_and_embs_list, load_text, load_bitexts


torch.multiprocessing.set_sharing_strategy("file_system")

logger = logging.getLogger(__name__)


def ebow_collator(batch: Union[list, dict]) -> dict:
    """Collate function for EmbBoW dataset"""

    # print("batch in collator:", len(batch), type(batch))

    if isinstance(batch, dict) and "rixs" in batch:
        batch_dict = batch
    else:
        batch_dict = {"rixs": []}
        for key in batch[0]:
            batch_dict[key] = []

        for i, sub_dict in enumerate(batch):  # list of dict
            # each dict corresponds to one doc
            rixs = torch.ones(len(sub_dict["vals"])).long() * i
            batch_dict["rixs"].append(rixs)
            for key, elem in sub_dict.items():
                batch_dict[key].append(elem)

        for key, lst in batch_dict.items():
            batch_dict[key] = torch.cat(lst, dim=0)

    return batch_dict


class ProcessedEmbBowDataset(Dataset):
    """A dataset to yield sentence embeddings along with BoW statistics from .npy .npz files"""

    def __init__(self, input_json):
        self.data_dict = {}
        with open(input_json, "r", encoding="utf-8") as fpr:
            self.data_dict = json.load(fpr)

        self.cur_key = None

        self.n_docs = 0
        self.key2ndocs = {}

        self.__determine_num_docs()

    def __determine_num_docs(self):
        for key, sub_dict in self.data_dict.items():
            self.key2ndocs[key] = scipy.sparse.load_npz(
                sub_dict["bow"]).shape[0]
            self.n_docs += self.key2ndocs[key]

    def __len__(self):
        return self.n_docs

    def __getitems__(self, ixs):
        pass


class IntraChunkSampler(Sampler[List[int]]):
    """Batch sampler to yield indices within each chunk"""

    def __init__(self, chunk_sizes: List[int], batch_size: int, shuffle: bool = False):
        self.chunk_sizes = chunk_sizes
        self.batch_size = batch_size  # , min(self.chunk_sizes))
        self.shuffle = shuffle

    def __len__(self):
        return self.batch_size

    def __iter__(self):
        offset = 0
        for csize in self.chunk_sizes:
            num_batches = max(1, int(csize // self.batch_size))
            # return batch indices for each chunk
            if self.shuffle:
                for ixs in torch.randperm(csize).chunk(num_batches):
                    yield ixs + offset
            else:
                yield from torch.arange(offset, offset + csize, dtype=torch.long).chunk(
                    num_batches
                )
            offset += csize


class EmbBoWDataset(Dataset):
    """A dataset to yield Sentence or Utterance Embeddings + BoW counts"""

    def __init__(
        self,
        list_text_file: list,
        list_emb1_file: list,
        list_emb2_file: list,
        list_sim_scores_file: list,
        cvect: CountVectorizer,
        sim_thresh: float = 0.8,
        norm: Union[None, str] = None,
        langs: Union[List, None] = None,
    ):
        # all_docs is list of documents
        # mmaps1, mmaps2 are list of memmaps of emb1 and emb2 respectively
        # sim_ixs is a nested list of indices that satisfy sim_thresh constraint
        all_docs, self.mmaps1, self.mmaps2, self.sim_ixs = filter_data_and_embs_list(
            list_text_file,
            list_emb1_file,
            list_emb2_file,
            list_sim_scores_file,
            sim_thresh=sim_thresh,
            norm=norm,
        )

        self.cvect = cvect
        self.langs = langs

        self.chunk_sizes = np.asarray([len(six) for six in self.sim_ixs])

        self.emb_dim = self.mmaps1[0].shape[1]

        logger.info("Building BoW matrix..")
        self.dbyw = self.cvect.transform(all_docs)

        logger.info(f"Total number of tokens: {self.dbyw.sum():,}")

        self.dbyw = self.dbyw.tocsr()
        self.n_docs, self.vocab_size = self.dbyw.shape

    def map_idx_to_memmap(self, idx):
        """map doc index to emb index based on chunk lengths"""

        if isinstance(idx, torch.Tensor):
            idx = idx.item()

        for i, cl in enumerate(self.chunk_sizes):
            if idx < cl:
                return i, self.sim_ixs[i][idx]
            idx -= cl

    def __len__(self):
        return self.n_docs

    def __getitem__(self, idx):
        batch_dbyw = self.dbyw[idx]
        # print("single dbyw", batch_dbyw.shape)
        rixs, cixs = batch_dbyw.nonzero()

        rixs = torch.from_numpy(rixs).long()
        cixs = torch.from_numpy(cixs).long()
        vals = torch.from_numpy(batch_dbyw.data).float()

        mmap_ix, eix = self.map_idx_to_memmap(idx)
        embs1 = np.array(self.mmaps1[mmap_ix][eix, :])
        embs2 = np.array(self.mmaps2[mmap_ix][eix, :])
        return {
            "cixs": cixs,
            "vals": vals,
            "doc_embs1": torch.from_numpy(embs1).float().view(1, -1),
            "doc_embs2": torch.from_numpy(embs2).float().view(1, -1),
        }

    def __getitems__(self, idxs):
        # print("idxs:", len(idxs))

        # ixs = torch.where(self.rixs == idxs)[0]

        # Convert torch tensor to numpy array for scipy compatibility
        if isinstance(idxs, torch.Tensor):
            idxs = idxs.numpy()

        batch_dbyw = self.dbyw[idxs]
        # print("batch dbyw", batch_dbyw.shape)
        rixs, cixs = batch_dbyw.nonzero()

        rixs = torch.from_numpy(rixs).long()
        cixs = torch.from_numpy(cixs).long()
        vals = torch.from_numpy(batch_dbyw.data).float()

        embs1 = torch.zeros(len(idxs), self.emb_dim)
        embs2 = torch.zeros(len(idxs), self.emb_dim)

        for i, idx in enumerate(idxs):
            mmap_ix, eix = self.map_idx_to_memmap(idx)
            embs1[i] = torch.from_numpy(np.array(self.mmaps1[mmap_ix][eix, :]))
            embs2[i] = torch.from_numpy(np.array(self.mmaps2[mmap_ix][eix, :]))

        return {
            "rixs": rixs,
            "cixs": cixs,
            "vals": vals,
            "doc_embs1": embs1.float(),
            "doc_embs2": embs2.float(),
        }


class TextBoWDataset(Dataset):
    """Text + BoW dataset"""

    def __init__(
        self,
        input_text_file,
        tokenizer,
        cvect,
        return_bow=False,
    ):
        self.input_text_file = input_text_file
        self.return_bow = return_bow

        input_sents = []
        with open(input_text_file, "r", encoding="utf-8") as fpr:
            for line in fpr:
                line = line.strip()
                if line:
                    input_sents.append(line)
        logger.info("Loaded input text with lines: %d", len(input_sents))

        self.cvect = cvect

        if return_bow:
            self.dbyw = self.cvect.transform(input_sents)
            self.dbyw = self.dbyw.tocsr()

            rixs, cixs = self.dbyw.nonzero()

            self.rixs = torch.from_numpy(rixs).long()
            self.cixs = torch.from_numpy(cixs).long()
            self.vals = torch.from_numpy(self.dbyw.data).float()
            self.vocab_size = self.dbyw.shape[1]

        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info("Tokenizing..")

        self.tokenized_sents = tokenizer(
            input_sents, padding=True, return_tensors="pt")

        self.n_sents = len(self.tokenized_sents["input_ids"])

    def __len__(self):
        return self.n_sents

    def __getitem__(self, idx):
        if self.return_bow:
            ixs = torch.where(self.rixs == idx)[0]
            ret_val = {
                "cixs": self.cixs[ixs],
                "vals": self.vals[ixs],
                "input_ids": self.tokenized_sents["input_ids"][idx].view(1, -1),
                "attention_mask": self.tokenized_sents["attention_mask"][idx].view(
                    1, -1
                ),
            }

        else:
            ret_val = {
                "input_ids": self.tokenized_sents["input_ids"][idx],
                "attention_mask": self.tokenized_sents["attention_mask"][idx],
            }
        return ret_val


class BoWDataset(Dataset):
    """Dataset for bag-of-words statistics"""

    def __init__(
        self,
        input_docs,
        cvect,
    ):
        self.cvect = cvect
        self.dbyw = cvect.transform(input_docs)
        self.n_docs, self.vocab_size = self.dbyw.shape

    def __len__(self):
        return self.n_docs

    def __getitem__(self, idx):
        batch_dbyw = self.dbyw[idx]
        # print("single dbyw", batch_dbyw.shape)
        rixs, cixs = batch_dbyw.nonzero()

        rixs = torch.from_numpy(rixs).long()
        cixs = torch.from_numpy(cixs).long()
        vals = torch.from_numpy(batch_dbyw.data).float()

        return {
            "rixs": rixs,
            "cixs": cixs,
            "vals": vals,
        }

    def __getitems__(self, idxs):
        batch_dbyw = self.dbyw[idxs]
        rixs, cixs = batch_dbyw.nonzero()

        rixs = torch.from_numpy(rixs).long()
        cixs = torch.from_numpy(cixs).long()
        vals = torch.from_numpy(batch_dbyw.data).float()

        return {
            "rixs": rixs,
            "cixs": cixs,
            "vals": vals,
        }


class TextDataset(Dataset):
    """Text dataset based on setences"""

    def __init__(self, text_fpaths: list, input_type: str = "flist"):
        assert isinstance(text_fpaths, list), "Should be list"

        self.texts = []

        if input_type == "flist":
            for fpath in text_fpaths:
                text = load_text(fpath)
                self.texts.extend(text)
        elif input_type == "text":
            self.texts = text_fpaths
        else:
            raise TypeError(
                "input_type should be either flist or text for TextDataset")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

    def __getitems__(self, indices):
        return [self.texts[idx] for idx in indices]


def bitext_collator(batch_list):
    """collator for BiTextDataset"""
    list1 = []
    list2 = []
    for tup in batch_list:
        list1.append(tup[0])
        list2.append(tup[1])
    return list1, list2


class BiTextDataset(Dataset):
    def __init__(self, text_fpath1, text_fpath2):
        self.texts1, self.texts2 = load_bitexts(text_fpath1, text_fpath2)

        assert len(self.texts1) == len(
            self.texts2
        ), f"Input texts have different number of lines {len(self.texts1)} != {len(self.texts2)}"

    def __len__(self):
        return len(self.texts1)

    def __getitem__(self, idx):
        return self.texts1[idx], self.texts2[idx]

    def __getitems__(self, indices):
        list_ = []
        for idx in indices:
            list_.append([self.texts1[idx], self.texts2[idx]])
        return list_


class SpeechTextDataset(Dataset):
    """paired speech and text dataset"""

    def __init__(self, audio_fpaths, text, min_dur=0.0, max_dur=80.0):
        self.audio_fpaths = audio_fpaths
        self.text = text
        self.min_dur = min_dur
        self.max_dur = max_dur

    def __len__(self):
        return len(self.audio_fpaths)

    def __getitems__(self, ixs):
        batch_audio = []
        batch_text = []
        batch_fpaths = []
        for i in ixs:
            fpath = self.audio_fpaths[i]
            raw_data, fs = torchaudio.load(fpath)  # pylint: disable=E1101
            if fs != 16000:
                re_sampler = torchaudio.transforms.Resample(fs, 16000)
                raw_data = re_sampler(raw_data)

            if self.min_dur <= raw_data.shape[1] / 16000 <= self.max_dur:
                batch_audio.append(raw_data)
                batch_text.append(self.text[i])
                batch_fpaths.append(fpath)
            else:
                logger.warning(
                    "Skipping %s due to duration %.2f", fpath, raw_data.shape[1] / 16000
                )

        return batch_audio, batch_text, batch_fpaths


def speech_text_collate_fn(batch_list):
    """Collate function for generating batches of re-sampled speech,
    and text from list of pairs or dict"""

    fpaths = []
    speech_data = []
    text_data = []

    if isinstance(batch_list[0], dict):
        logger.error("No path to audio. Exiting")
        sys.exit()

        for sub_dict in batch_list:
            raw_data = torch.from_numpy(sub_dict["audio"]["array"]).float()
            fs = sub_dict["audio"]["sampling_rate"]
            if fs != 16000:
                re_sampler = torchaudio.transforms.Resample(fs, 16000)
                raw_data = re_sampler(raw_data)

            speech_data.append(raw_data)
            text_data.append(sub_dict["sentence"])

    else:
        speech_data = batch_list[0]
        text_data = batch_list[1]
        fpaths = batch_list[2]

    return speech_data, text_data, fpaths


class SpeechTripletDataset(Dataset):
    """Speech, text, translation triplet dataset for MUST-C
    Takes segment info in the form of dictionary and text and translation"""

    def __init__(
        self, audio_dir, seg_data, src_texts, tgt_texts, min_dur=0.0, max_dur=80.0
    ):
        assert (
            len(seg_data) == len(src_texts) == len(tgt_texts)
        ), f"Input data have different number of lines {len(seg_data)} != {len(src_texts)} != {len(tgt_texts)}"
        self.audio_dir = audio_dir
        self.seg_data = seg_data
        self.min_dur = min_dur
        self.max_dur = max_dur
        self.wav2segs = self.__get_wav2segs(src_texts, tgt_texts)
        self.chunk_sizes = [len(self.wav2segs[wav]) for wav in self.wav2segs]

    def __get_wav2segs(self, src_texts, tgt_texts):
        """Get wav 2 segments mapping"""
        wav2segs = {}
        for i, row in enumerate(self.seg_data):
            wav = row["wav"]
            row["src_text"] = src_texts[i]
            row["tgt_text"] = tgt_texts[i]
            row["audio_fpath"] = os.path.join(self.audio_dir, wav)
            if wav not in wav2segs:
                wav2segs[wav] = []
            wav2segs[wav].append(row)
        return wav2segs

    def __len__(self):
        return len(self.seg_data)

    def map_to_index_within_chunk(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        for cix, chunk_size in enumerate(self.chunk_sizes):
            if idx < chunk_size:
                return idx
            idx -= chunk_size

    def __getitems__(self, ixs):
        # all ixs correspond to same wav_id because of IntraChunkSampler
        # ixs can range anwhere from 0 to len(seg_data) -> total num of segments

        # print("ixs", ixs)

        audio_fpath = self.seg_data[ixs[0]]["audio_fpath"]
        audio, fs = torchaudio.load(audio_fpath)
        # resample to 16kHz
        audio = torchaudio.transforms.Resample(
            orig_freq=fs, new_freq=16000)(audio)

        batch_segs = []
        batch_src_texts = []
        batch_tgt_texts = []
        batch_meta = []
        # extract the segments from the audio
        for i in ixs:
            seg = self.seg_data[i]
            # cix = self.map_to_index_within_chunk(i)

            offset = float(seg["offset"])
            duration = float(seg["duration"])
            if self.min_dur <= duration <= self.max_dur:
                segment = audio[
                    :, int(offset * 16000): int((offset + duration) * 16000)
                ]

                batch_segs.append(segment)
                batch_src_texts.append(seg["src_text"])
                batch_tgt_texts.append(seg["tgt_text"])
                batch_meta.append(json.dumps(seg, ensure_ascii=False))
            else:
                logger.warning(
                    "Skipping segment %s due to duration %.2f",
                    json.dumps(seg, ensure_ascii=False),
                    duration,
                )

        return batch_segs, batch_src_texts, batch_tgt_texts, batch_meta


def speech_triplet_collate_fn(batch_list):
    """Collate function for generating batches of re-sampled speech,
    text, and translations from list of pairs or dict"""

    return batch_list
