def files_type(fs0, fs1, files):
    """Inspects the file type of the given files."""
    for file_meta in files['deleted_files']:
        file_meta['type'] = fs0.file(file_meta['path'])
    for file_meta in files['created_files'] + files['modified_files']:
        file_meta['type'] = fs1.file(file_meta['path'])

    return files