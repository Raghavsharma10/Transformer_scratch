def main(argv):
    """Call startup.call() in your main()."""
    args = startup.call(argv=argv)['args']
    if args.v:
        print('x * y = %d' % (args.x * args.y))
    else:
        print(args.x * args.y)
    return 0