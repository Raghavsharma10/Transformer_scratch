def default_parser() -> argparse.ArgumentParser:
    """Create a parser for CLI arguments and options."""
    parser = argparse.ArgumentParser(
        prog=CONSOLE_SCRIPT,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    build_parser(parser)
    return parser