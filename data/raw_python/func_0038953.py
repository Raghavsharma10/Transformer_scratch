def cli_run():
    """
    Command line interface

    This interface saves you coding effort to:

        - display basic info (classes, sizes etc) about datasets
        - display meta data (class membership) for samples
        - perform basic arithmetic (add multiple classes or feature sets)


    """

    path_list, meta_requested, summary_requested, add_path_list, out_path = parse_args()

    # printing info if requested
    if path_list:
        for ds_path in path_list:
            ds = MLDataset(ds_path)
            if summary_requested:
                print_info(ds, ds_path)
            if meta_requested:
                print_meta(ds, ds_path)

    # combining datasets
    if add_path_list:
        combine_and_save(add_path_list, out_path)

    return