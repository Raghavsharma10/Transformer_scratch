"""
tokenizer/train_tokenizer.py
Train the BPE tokenizer on Python source files.

Usage:
    python -m tokenizer.train_tokenizer \
        --data_dir  data/raw_python \
        --output    tokenizer/python_bpe \
        --vocab_size 8000
"""

import argparse
import glob
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Python BPE tokenizer")
    parser.add_argument("--data_dir",   default="data/raw_python",
                        help="Directory containing .py training files")
    parser.add_argument("--output",     default="tokenizer/python_bpe",
                        help="Model prefix (will create .model and .vocab files)")
    parser.add_argument("--vocab_size", type=int, default=8000,
                        help="BPE vocabulary size (default: 8000)")
    parser.add_argument("--coverage",  type=float, default=0.9999,
                        help="Character coverage (default: 0.9999)")
    parser.add_argument("--max_files", type=int, default=None,
                        help="Cap number of training files (for quick tests)")
    args = parser.parse_args()

    # Collect .py files
    pattern = os.path.join(args.data_dir, "**", "*.py")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        print(f"[ERROR] No .py files found under {args.data_dir}")
        sys.exit(1)

    if args.max_files:
        files = files[: args.max_files]

    print(f"Found {len(files)} Python files for tokenizer training.")

    from tokenizer.bpe import PythonBPETokenizer
    tok = PythonBPETokenizer()
    tok.train(
        input_files=files,
        vocab_size=args.vocab_size,
        model_prefix=args.output,
        character_coverage=args.coverage,
    )

    # Quick sanity check
    snippet = "def hello(name: str) -> None:\n    print(f'hello {name}')\n"
    ids = tok.encode(snippet)
    decoded = tok.decode(ids)
    print(f"\nSanity check:")
    print(f"  Input  : {repr(snippet[:60])}")
    print(f"  Ids    : {ids[:20]} ...")
    print(f"  Decoded: {repr(decoded[:60])}")
    print(f"\nTokenizer saved to {args.output}.model")


if __name__ == "__main__":
    main()
