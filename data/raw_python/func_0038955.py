def print_meta(ds, ds_path=None):
    "Prints meta data for subjects in given dataset."

    print('\n#' + ds_path)
    for sub, cls in ds.classes.items():
        print('{},{}'.format(sub, cls))

    return