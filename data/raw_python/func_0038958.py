def parse_args():
    """Arg parser.

    """

    parser = get_parser()

    if len(sys.argv) < 2:
        parser.print_help()
        logging.warning('Too few arguments!')
        parser.exit(1)

    # parsing
    try:
        params = parser.parse_args()
    except Exception as exc:
        print(exc)
        raise ValueError('Unable to parse command-line arguments.')

    path_list = list()
    if params.path_list is not None:
        for dpath in params.path_list:
            if pexists(dpath):
                path_list.append(realpath(dpath))
            else:
                print('Below dataset does not exist. Ignoring it.\n{}'.format(dpath))

    add_path_list = list()
    out_path = None
    if params.add_path_list is not None:
        for dpath in params.add_path_list:
            if pexists(dpath):
                add_path_list.append(realpath(dpath))
            else:
                print('Below dataset does not exist. Ignoring it.\n{}'.format(dpath))

        if params.out_path is None:
            raise ValueError(
                'Output path must be specified to save the combined dataset to')

        out_path = realpath(params.out_path)
        parent_dir = dirname(out_path)
        if not pexists(parent_dir):
            os.mkdir(parent_dir)

        if len(add_path_list) < 2:
            raise ValueError('Need a minimum of datasets to combine!!')

    # removing duplicates (from regex etc)
    path_list = set(path_list)
    add_path_list = set(add_path_list)

    return path_list, params.meta_requested, params.summary_requested, \
           add_path_list, out_path