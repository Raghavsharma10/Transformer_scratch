"""
tokenizer/bpe.py
BPE tokenizer with Python-aware special tokens.
Wraps SentencePiece for sub-word tokenization while preserving
indentation semantics crucial for Python code.
"""

from __future__ import annotations

import os
import re
import tokenize
import io
from pathlib import Path
from typing import List, Optional

try:
    import sentencepiece as spm
    _SPM_AVAILABLE = True
except ImportError:
    _SPM_AVAILABLE = False


# ──────────────────────────────────────────────
# Special token registry
# ──────────────────────────────────────────────

SPECIAL_TOKENS: dict[str, int] = {
    "<PAD>":     0,
    "<BOF>":     1,   # beginning of file / context
    "<EOF>":     2,
    "<UNK>":     3,
    "<INDENT>":  4,   # Python indentation level increase
    "<DEDENT>":  5,   # Python indentation level decrease
    "<NEWLINE>": 6,   # logical newline (vs. continuation)
    "<COMMENT>": 7,   # stripped inline comment placeholder
}

PAD_ID     = SPECIAL_TOKENS["<PAD>"]
BOF_ID     = SPECIAL_TOKENS["<BOF>"]
EOF_ID     = SPECIAL_TOKENS["<EOF>"]
UNK_ID     = SPECIAL_TOKENS["<UNK>"]
INDENT_ID  = SPECIAL_TOKENS["<INDENT>"]
DEDENT_ID  = SPECIAL_TOKENS["<DEDENT>"]
NEWLINE_ID = SPECIAL_TOKENS["<NEWLINE>"]


# ──────────────────────────────────────────────
# Indentation normaliser
# ──────────────────────────────────────────────

def normalise_indentation(source: str) -> str:
    """
    Replace Python INDENT/DEDENT tokens (as emitted by the tokenize module)
    with literal placeholder strings so BPE can treat them as atomic units.
    Falls back to a simple regex approach if the source is not valid Python.
    """
    placeholder_indent  = " __INDENT__ "
    placeholder_dedent  = " __DEDENT__ "
    placeholder_newline = "\n__NEWLINE__\n"

    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except tokenize.TokenError:
        # Not valid Python (e.g. incomplete snippet) — pass through unchanged
        return source

    out_parts: list[str] = []
    prev_end = (1, 0)

    for tok in tokens:
        ttype, tstr, start, end, _ = tok
        if ttype == tokenize.INDENT:
            out_parts.append(placeholder_indent)
        elif ttype == tokenize.DEDENT:
            out_parts.append(placeholder_dedent)
        elif ttype == tokenize.NEWLINE or ttype == tokenize.NL:
            out_parts.append(placeholder_newline)
        elif ttype == tokenize.ENDMARKER:
            break
        else:
            out_parts.append(tstr)
        prev_end = end

    normalised = " ".join(out_parts)
    return normalised


def denormalise_indentation(text: str, indent_width: int = 4) -> str:
    """Reverse the normalisation — reconstruct Python indentation."""
    indent_level = 0
    lines = text.replace("__NEWLINE__", "\n").split("\n")
    result: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped == "__INDENT__":
            indent_level += 1
            continue
        if stripped == "__DEDENT__":
            indent_level = max(0, indent_level - 1)
            continue
        if stripped:
            result.append(" " * (indent_level * indent_width) + stripped)
    return "\n".join(result)


# ──────────────────────────────────────────────
# Tokenizer class
# ──────────────────────────────────────────────

class PythonBPETokenizer:
    """
    BPE tokenizer tuned for Python source code.

    Usage:
        tok = PythonBPETokenizer()
        tok.train(corpus_files, vocab_size=8000, model_prefix="tokenizer/python_bpe")
        ids = tok.encode("def foo():\n    return 42")
        text = tok.decode(ids)
    """

    INDENT_PLACEHOLDER  = "__INDENT__"
    DEDENT_PLACEHOLDER  = "__DEDENT__"
    NEWLINE_PLACEHOLDER = "__NEWLINE__"

    def __init__(self, model_path: Optional[str] = None):
        self.sp_model: Optional[spm.SentencePieceProcessor] = None
        self.vocab_size: int = 0
        # Map placeholder strings → special token IDs
        self._placeholder_to_id = {
            self.INDENT_PLACEHOLDER:  INDENT_ID,
            self.DEDENT_PLACEHOLDER:  DEDENT_ID,
            self.NEWLINE_PLACEHOLDER: NEWLINE_ID,
        }
        self._id_to_placeholder = {v: k for k, v in self._placeholder_to_id.items()}

        if model_path and os.path.exists(model_path):
            self.load(model_path)

    # ── Training ────────────────────────────────

    def train(
        self,
        input_files: List[str],
        vocab_size: int = 8000,
        model_prefix: str = "tokenizer/python_bpe",
        character_coverage: float = 0.9999,
    ) -> None:
        if not _SPM_AVAILABLE:
            raise ImportError("sentencepiece is required: pip install sentencepiece")

        os.makedirs(os.path.dirname(model_prefix) or ".", exist_ok=True)

        # Write normalised corpus to a temp file
        norm_corpus = model_prefix + "_norm_corpus.txt"
        with open(norm_corpus, "w", encoding="utf-8") as f:
            for fpath in input_files:
                with open(fpath, encoding="utf-8", errors="ignore") as src:
                    text = src.read()
                norm = normalise_indentation(text)
                f.write(norm + "\n")

        # SentencePiece training
        spm.SentencePieceTrainer.train(
            input=norm_corpus,
            model_prefix=model_prefix,
            vocab_size=vocab_size - len(SPECIAL_TOKENS),
            character_coverage=character_coverage,
            model_type="bpe",
            pad_id=PAD_ID,
            unk_id=UNK_ID,
            bos_id=BOF_ID,
            eos_id=EOF_ID,
            user_defined_symbols=list(SPECIAL_TOKENS.keys()),
            split_digits=True,          # treat digits individually
            byte_fallback=True,         # handle rare Unicode safely
        )

        self.load(model_prefix + ".model")
        os.remove(norm_corpus)
        print(f"Tokenizer trained. Vocab size: {self.vocab_size}")

    # ── Load / save ─────────────────────────────

    def load(self, model_path: str) -> None:
        if not _SPM_AVAILABLE:
            raise ImportError("sentencepiece is required: pip install sentencepiece")
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(model_path)
        self.vocab_size = self.sp_model.get_piece_size()

    def save(self, model_prefix: str) -> None:
        """Already persisted by SentencePiece; this is a no-op placeholder."""
        pass

    # ── Encode / decode ─────────────────────────

    def encode(
        self,
        text: str,
        add_bof: bool = True,
        add_eof: bool = False,
        max_length: Optional[int] = None,
    ) -> List[int]:
        if self.sp_model is None:
            raise RuntimeError("Tokenizer not loaded. Call train() or load() first.")

        norm = normalise_indentation(text)
        ids: List[int] = self.sp_model.encode(norm)

        if add_bof:
            ids = [BOF_ID] + ids
        if add_eof:
            ids = ids + [EOF_ID]
        if max_length is not None:
            ids = ids[:max_length]
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        if self.sp_model is None:
            raise RuntimeError("Tokenizer not loaded. Call train() or load() first.")

        if skip_special:
            ids = [i for i in ids if i not in SPECIAL_TOKENS.values()]

        text = self.sp_model.decode(ids)
        return denormalise_indentation(text)

    def token_to_id(self, token: str) -> int:
        if token in SPECIAL_TOKENS:
            return SPECIAL_TOKENS[token]
        return self.sp_model.piece_to_id(token) if self.sp_model else UNK_ID

    def id_to_token(self, id_: int) -> str:
        if id_ in self._id_to_placeholder:
            return self._id_to_placeholder[id_]
        return self.sp_model.id_to_piece(id_) if self.sp_model else "<UNK>"

    def __len__(self) -> int:
        return self.vocab_size if self.vocab_size else len(SPECIAL_TOKENS)


# ──────────────────────────────────────────────
# Fallback character-level tokenizer (no deps)
# ──────────────────────────────────────────────

class CharTokenizer:
    """
    Simple character-level tokenizer — no external dependencies.
    Use when SentencePiece is unavailable or for quick smoke tests.
    """

    def __init__(self) -> None:
        self.char_to_id: dict[str, int] = dict(SPECIAL_TOKENS)
        self.id_to_char: dict[int, str] = {v: k for k, v in SPECIAL_TOKENS.items()}
        self.vocab_size = len(SPECIAL_TOKENS)

    def build_vocab(self, corpus: str) -> None:
        for ch in sorted(set(corpus)):
            if ch not in self.char_to_id:
                self.char_to_id[ch] = self.vocab_size
                self.id_to_char[self.vocab_size] = ch
                self.vocab_size += 1

    def encode(self, text: str, add_bof: bool = True, add_eof: bool = False,
               max_length: Optional[int] = None) -> List[int]:
        ids = [self.char_to_id.get(c, UNK_ID) for c in text]
        if add_bof:
            ids = [BOF_ID] + ids
        if add_eof:
            ids = ids + [EOF_ID]
        if max_length is not None:
            ids = ids[:max_length]
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        if skip_special:
            ids = [i for i in ids if i not in SPECIAL_TOKENS.values()]
        return "".join(self.id_to_char.get(i, "?") for i in ids)

    def __len__(self) -> int:
        return self.vocab_size
