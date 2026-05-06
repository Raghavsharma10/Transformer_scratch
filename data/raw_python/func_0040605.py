def open_filezip(file_path, find_str):
    """
    Open the wrapped file.
    Read directly from the zip without extracting its content.
    """
    if zipfile.is_zipfile(file_path):
        zipf = zipfile.ZipFile(file_path)
        interesting_files = [f for f in zipf.infolist() if find_str in f]

        for inside_file in interesting_files:
            yield zipf.open(inside_file)