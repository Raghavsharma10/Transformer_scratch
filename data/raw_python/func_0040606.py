def extract_filezip(path_to_file, dest_path, target_zipfiles=None):
    """
    Extract file zip to destiny path folder targeting only some kind of files.
    """

    target_zipfiles = ['.*'] if target_zipfiles is None else target_zipfiles

    files = []
    _, ext = os.path.splitext(path_to_file)

    if ext == '.zip':
        file = open(path_to_file, 'rb')
        with zipfile.ZipFile(file) as zip_file:
            regexp = '|'.join(target_zipfiles) if target_zipfiles else '.*'
            search_regex = re.compile(regexp)

            lista = [m.group() for x in zip_file.namelist()
                     for m in [search_regex.search(x)] if m]

            for zp_file in lista:
                try:
                    zip_file.extract(zp_file, dest_path)
                    files.append(os.path.join(dest_path, zp_file))
                except Exception as ex:
                    msg = 'Fail to extract {} in {} to {} - {}'.format(
                        zp_file, path_to_file, dest_path, ex)
                    logger.error(msg)
        file.close()
    else:
        logger.warning('Not zipfile passed in args')
    return files