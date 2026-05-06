def format_installed_dap(name, full=False):
    '''Formats information about an installed DAP in a human readable form to list of lines'''
    dap_data = get_installed_daps_detailed().get(name)
    if not dap_data:
        raise DapiLocalError('DAP "{dap}" is not installed, can not query for info.'.format(dap=name))

    locations = [os.path.join(data['location'], '') for data in dap_data]
    for location in locations:
        dap = dapi.Dap(None, fake=True, mimic_filename=name)
        meta_path = os.path.join(location, 'meta', name + '.yaml')
        with open(meta_path, 'r') as fh:
            dap.meta = dap._load_meta(fh)
        dap.files = _get_assistants_snippets(location, name)
        dap._find_bad_meta()

        format_local_dap(dap, full=full, custom_location=os.path.dirname(location))