def find_local_changes():
    """ Find things that have changed since the last run, applying ignore filters """

    manifest = data_store.read_local_manifest()
    old_state = manifest['files']
    current_state = get_file_list(config['data_dir'])
    current_state = [fle for fle in current_state if not
                     next((True for flter in config['ignore_filters']
                           if fnmatch.fnmatch(fle['path'], flter)), False)]
    return manifest, find_manifest_changes(current_state, old_state)