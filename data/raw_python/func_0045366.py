def main():
    """Check arguments are retrieved."""
    args = parse_arguments()
    print(args)
    print("action" + args.action)
    print("pid" + args.pid)
    print("directory" + args.directory)
    print("fileprefix" + args.fileprefix)