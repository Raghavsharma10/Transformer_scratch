def upload_scripts(client, script_dir, overwrite=True):
    """Uploads general-purpose scripts to a Google Storage bucket.
    
    TODO: docstring"""
    local_dir = os.path.join(genometools._root, 'data', 'gcloud', 'scripts')
    match = _BUCKET_PAT.match(script_dir)
    script_bucket = match.group(1)
    script_prefix = match.group(2)

    depth = len(local_dir.split(os.sep))

    for root, dirs, files in os.walk(local_dir):
        rel_path = '/'.join(root.split(os.sep)[depth:])
        for f in files:
            local_path = os.path.join(root, f)

            if rel_path:
                remote_path = '/'.join([script_prefix, rel_path, f])
            else:
                remote_path = '/'.join([script_prefix, f])
            _LOGGER.info('Uploading "%s"...', remote_path)
            storage.upload_file(client, script_bucket, local_path, remote_path,
                                overwrite=overwrite)