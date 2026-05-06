def _download_and_decompress_if_necessary(
        full_path,
        download_url,
        timeout=None,
        use_wget_if_available=False):
    """
    Downloads remote file at `download_url` to local file at `full_path`
    """
    logger.info("Downloading %s to %s", download_url, full_path)
    filename = os.path.split(full_path)[1]
    base_name, ext = os.path.splitext(filename)
    tmp_path = _download_to_temp_file(
        download_url=download_url,
        timeout=timeout,
        base_name=base_name,
        ext=ext,
        use_wget_if_available=use_wget_if_available)

    if download_url.endswith("zip") and not filename.endswith("zip"):
        logger.info("Decompressing zip into %s...", filename)
        with zipfile.ZipFile(tmp_path) as z:
            names = z.namelist()
            assert len(names) > 0, "Empty zip archive"
            if filename in names:
                chosen_filename = filename
            else:
                # If zip archive contains multiple files, choose the biggest.
                biggest_size = 0
                chosen_filename = names[0]
                for info in z.infolist():
                    if info.file_size > biggest_size:
                        chosen_filename = info.filename
                        biggest_size = info.file_size
            extract_path = z.extract(chosen_filename)
        move(extract_path, full_path)
        os.remove(tmp_path)
    elif download_url.endswith("gz") and not filename.endswith("gz"):
        logger.info("Decompressing gzip into %s...", filename)
        with gzip.GzipFile(tmp_path) as src:
            contents = src.read()
        os.remove(tmp_path)
        with open(full_path, 'wb') as dst:
            dst.write(contents)
    elif download_url.endswith(("html", "htm")) and full_path.endswith(".csv"):
        logger.info("Extracting HTML table into CSV %s...", filename)
        df = pd.read_html(tmp_path, header=0)[0]
        df.to_csv(full_path, sep=',', index=False, encoding='utf-8')
    else:
        move(tmp_path, full_path)