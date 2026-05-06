def get_bundle_ref(args, l, use_history=False):
    """ Use a variety of methods to determine which bundle to use

    :param args:
    :return:
    """

    if not use_history:

        if args.id:
            return (args.id, '-i argument')

        if hasattr(args, 'bundle_ref') and args.bundle_ref:
            return (args.bundle_ref, 'bundle_ref argument')


        if 'AMBRY_BUNDLE' in os.environ:
            return (os.environ['AMBRY_BUNDLE'], 'environment')

        cwd_bundle = os.path.join(os.getcwd(), 'bundle.yaml')

        if os.path.exists(cwd_bundle):

            with open(cwd_bundle) as f:
                from ambry.identity import Identity

                config = yaml.load(f)
                try:
                    ident = Identity.from_dict(config['identity'])
                    return (ident.vid, 'directory')
                except KeyError:
                    pass

    history = l.edit_history()

    if history:
        return (history[0].d_vid, 'history')

    return None, None