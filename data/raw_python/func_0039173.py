def extract_files(filesystem, files, path):
    """Extracts requested files.

    files must be a list of files in the format

        {"C:\\Windows\\System32\\NTUSER.DAT": "sha1_hash"} for windows
        {"/home/user/text.txt": "sha1_hash"} for other FS.

    files will be extracted into path which must exist beforehand.

    Returns two dictionaries:

        {"sha1": "/local/path/sha1"} files successfully extracted
        {"sha1": "C:\\..\\text.txt"} files which could not be extracted windows
        {"sha1": "/../text.txt"} files which could not be extracted linux

    """
    extracted_files = {}
    failed_extractions = {}

    for file_to_extract in files:
        source = file_to_extract['path']
        destination = Path(path, file_to_extract['sha1'])

        if not destination.exists():
            destination = str(destination)

            try:
                filesystem.download(source, destination)
                extracted_files[file_to_extract['sha1']] = destination
            except RuntimeError:
                failed_extractions[file_to_extract['sha1']] = source
        else:
            extracted_files[file_to_extract['sha1']] = destination

    return extracted_files, failed_extractions