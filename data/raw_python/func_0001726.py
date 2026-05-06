def main() -> None:
    """Main entry point for console commands."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json",
        help="output in JSON format",
        action="store_true",
        default=False)
    parser.add_argument(
        "--config-file", help="Select config file to use", default=".snekrc")
    parser.add_argument(
        'files',
        metavar='file',
        nargs='*',
        default=[],
        help='Files to run checks against')
    parser.add_argument(
        "--init", help="generate snekrc", action="store_true", default=False)

    args = parser.parse_args()

    run_main(args)