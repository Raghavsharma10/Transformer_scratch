def create(asset_dir, dest_dir):
    """Command line call to create an asset zip

    :param asset_dir: (full path to the asset dir)
    :param dest_dir: (full path to the destination directory)
    :return: (int)
    """
    val = validate(asset_dir=asset_dir)
    if val != 0:
        return 1
    try:
        asset_zip = make_asset_zip(asset_dir_path=asset_dir, destination_directory=dest_dir)
    except AssetZipCreationError:
        _, ex, trace = sys.exc_info()
        msg = 'AssetZipCreationError: Problem with asset zip creation\n{e}'.format(e=str(ex))
        print('ERROR: {m}'.format(m=msg))
        return 1
    print('Created asset zip file: {z}'.format(z=asset_zip))
    return 0