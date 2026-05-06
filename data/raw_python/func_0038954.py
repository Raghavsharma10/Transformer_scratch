def print_info(ds, ds_path=None):
    "Prints basic summary of a given dataset."

    if ds_path is None:
        bname = ''
    else:
        bname = basename(ds_path)

    dashes = '-' * len(bname)
    print('\n{}\n{}\n{:full}'.format(dashes, bname, ds))

    return