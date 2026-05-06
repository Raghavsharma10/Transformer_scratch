def main():
    """The main entry point for command line invocation. It's output
    is adjusted by command line arguments. By default it outputs 10
    passwords.

    For help on accepted arguments, run::

        $ passgen -h

    Or::

        $ python -m passgen -h
    """
    parser = argparse.ArgumentParser(
        description="Generate random password."
    )
    parser.add_argument("-l", "--length",
                        help="the length of the generated "
                             "password (default: 12)",
                        type=int, default=12)
    parser.add_argument("-n", "--number",
                        help="how many passwords to generate (default: 10)",
                        type=int, default=10)
    parser.add_argument("-p", "--punctuation",
                        help="use punctuation characters",
                        action='store_true')
    parser.add_argument("--limit-punctuation",
                        help="specify allowed punctuation characters",
                        action='store', default='')
    alnum_group = parser.add_mutually_exclusive_group()
    alnum_group.add_argument("--no-digits",
                             help="don't use digits",
                             action='store_false', dest='digits')
    alnum_group.add_argument("--no-letters",
                             help="don't use letters",
                             action='store_false', dest='letters')
    case_group = parser.add_mutually_exclusive_group()
    case_group.add_argument("--upper",
                            help="use only upper case letters",
                            action='store_true')
    case_group.add_argument("--lower",
                            help="use only lower case letters",
                            action='store_true')
    args = parser.parse_args()

    if args.length < 1:
        _error("argument -l/--length must be greater than zero")
    if args.number < 1:
        _error("argument -n/--number must be greater than zero")

    if args.lower:
        case = "lower"
    elif args.upper:
        case = "upper"
    else:
        case = "both"

    for _ in range(args.number):
        print(passgen(args.length, punctuation=args.punctuation,
                      limit_punctuation=args.limit_punctuation,
                      digits=args.digits,
                      letters=args.letters, case=case))