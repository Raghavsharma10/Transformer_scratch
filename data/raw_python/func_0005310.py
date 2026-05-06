def validate(asset_dir):
    """Command line call to validate an asset structure

    :param asset_dir: (full path to the asset dir)
    :return: (int)
    """
    try:
        asset_name = validate_asset_structure(asset_dir_path=asset_dir)
    except Cons3rtAssetStructureError:
        _, ex, trace = sys.exc_info()
        msg = 'Cons3rtAssetStructureError: Problem with asset validation\n{e}'.format(e=str(ex))
        print('ERROR: {m}'.format(m=msg))
        return 1
    print('Validated asset with name: {n}'.format(n=asset_name))
    return 0