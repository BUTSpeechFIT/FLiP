"""
LINT: Log-Linear Interpretable Models for Multilingual Embeddings

A research codebase for learning interpretable word embeddings from
multilingual, multimodal sentence embeddings (primarily SONAR).
"""

__version__ = "0.1.0"
__author__ = "BUTSpeechFIT"

from lolm.models.interpretable import LoLM, FactLoLM, BayLoLM

__all__ = ["LoLM", "FactLoLM", "BayLoLM"]
