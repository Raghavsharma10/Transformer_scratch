def module(args):
    " Copy module source to current directory. "

    mod = op.join(settings.MOD_DIR, args.MODULE)
    assert op.exists(mod), "Not found module: %s" % args.MODULE
    if not args.DEST.startswith(op.sep):
        args.DEST = op.join(getcwd(), args.DEST)
    print_header("Copy module source")
    copytree(mod, args.DEST)