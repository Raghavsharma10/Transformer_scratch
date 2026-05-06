def characterize_local_files(filedir, max_bytes=MAX_FILE_DEFAULT):
    """
    Collate local file info as preperation for Open Humans upload.

    Note: Files with filesize > max_bytes are not included in returned info.

    :param filedir: This field is target directory to get files from.
    :param max_bytes: This field is the maximum file size to consider. Its
        default value is 128m.
    """
    file_data = {}
    logging.info('Characterizing files in {}'.format(filedir))
    for filename in os.listdir(filedir):
        filepath = os.path.join(filedir, filename)
        file_stats = os.stat(filepath)
        creation_date = arrow.get(file_stats.st_ctime).isoformat()
        file_size = file_stats.st_size
        if file_size <= max_bytes:
            file_md5 = hashlib.md5()
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    file_md5.update(chunk)
            md5 = file_md5.hexdigest()
            file_data[filename] = {
                'tags': guess_tags(filename),
                'description': '',
                'md5': md5,
                'creation_date': creation_date,
            }
    return file_data