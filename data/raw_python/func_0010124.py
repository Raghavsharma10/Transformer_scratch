def split_input(args):
    """Split query input into local files and URLs."""
    args['files'] = []
    args['urls'] = []
    for arg in args['query']:
        if os.path.isfile(arg):
            args['files'].append(arg)
        else:
            args['urls'].append(arg.strip('/'))