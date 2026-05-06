def import_file(
    src,
    file_name,
    imported_format=ImportedFormat.BOTH,
    progress_from=0.0,
    progress_to=None,
):
    """Import file to working directory.

    :param src: Source file path or URL
    :param file_name: Source file name
    :param imported_format: Import file format (extracted, compressed or both)
    :param progress_from: Initial progress value
    :param progress_to: Final progress value
    :return: Destination file path (if extracted and compressed, extracted path given)
    """

    if progress_to is not None:
        if not isinstance(progress_from, float) or not isinstance(progress_to, float):
            raise ValueError("Progress_from and progress_to must be float")

        if progress_from < 0 or progress_from > 1:
            raise ValueError("Progress_from must be between 0 and 1")

        if progress_to < 0 or progress_to > 1:
            raise ValueError("Progress_to must be between 0 and 1")

        if progress_from >= progress_to:
            raise ValueError("Progress_to must be higher than progress_from")

    print("Importing and compressing {}...".format(file_name))

    def importGz():
        """Import gzipped file.

        The file_name must have .gz extension.
        """
        if imported_format != ImportedFormat.COMPRESSED:  # Extracted file required
            with open(file_name[:-3], 'wb') as f_out, gzip.open(src, 'rb') as f_in:
                try:
                    shutil.copyfileobj(f_in, f_out, CHUNK_SIZE)
                except zlib.error:
                    raise ValueError("Invalid gzip file format: {}".format(file_name))

        else:  # Extracted file not-required
            # Verify the compressed file.
            with gzip.open(src, 'rb') as f:
                try:
                    while f.read(CHUNK_SIZE) != b'':
                        pass
                except zlib.error:
                    raise ValueError("Invalid gzip file format: {}".format(file_name))

        if imported_format != ImportedFormat.EXTRACTED:  # Compressed file required
            try:
                shutil.copyfile(src, file_name)
            except shutil.SameFileError:
                pass  # Skip copy of downloaded files

        if imported_format == ImportedFormat.COMPRESSED:
            return file_name
        else:
            return file_name[:-3]

    def import7z():
        """Import compressed file in various formats.

        Supported extensions: .bz2, .zip, .rar, .7z, .tar.gz, and .tar.bz2.
        """
        extracted_name, _ = os.path.splitext(file_name)
        destination_name = extracted_name
        temp_dir = 'temp_{}'.format(extracted_name)

        cmd = '7z x -y -o{} {}'.format(shlex.quote(temp_dir), shlex.quote(src))
        try:
            subprocess.check_call(cmd, shell=True)
        except subprocess.CalledProcessError as err:
            if err.returncode == 2:
                raise ValueError("Failed to extract file: {}".format(file_name))
            else:
                raise

        paths = os.listdir(temp_dir)
        if len(paths) == 1 and os.path.isfile(os.path.join(temp_dir, paths[0])):
            # Single file in archive.
            temp_file = os.path.join(temp_dir, paths[0])

            if imported_format != ImportedFormat.EXTRACTED:  # Compressed file required
                with open(temp_file, 'rb') as f_in, gzip.open(
                    extracted_name + '.gz', 'wb'
                ) as f_out:
                    shutil.copyfileobj(f_in, f_out, CHUNK_SIZE)

            if imported_format != ImportedFormat.COMPRESSED:  # Extracted file required
                shutil.move(temp_file, './{}'.format(extracted_name))

                if extracted_name.endswith('.tar'):
                    with tarfile.open(extracted_name) as tar:
                        tar.extractall()

                    os.remove(extracted_name)
                    destination_name, _ = os.path.splitext(extracted_name)
            else:
                destination_name = extracted_name + '.gz'
        else:
            # Directory or several files in archive.
            if imported_format != ImportedFormat.EXTRACTED:  # Compressed file required
                with tarfile.open(extracted_name + '.tar.gz', 'w:gz') as tar:
                    for fname in glob.glob(os.path.join(temp_dir, '*')):
                        tar.add(fname, os.path.basename(fname))

            if imported_format != ImportedFormat.COMPRESSED:  # Extracted file required
                for path in os.listdir(temp_dir):
                    shutil.move(os.path.join(temp_dir, path), './{}'.format(path))
            else:
                destination_name = extracted_name + '.tar.gz'

        shutil.rmtree(temp_dir)
        return destination_name

    def importUncompressed():
        """Import uncompressed file."""
        if imported_format != ImportedFormat.EXTRACTED:  # Compressed file required
            with open(src, 'rb') as f_in, gzip.open(file_name + '.gz', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out, CHUNK_SIZE)

        if imported_format != ImportedFormat.COMPRESSED:  # Extracted file required
            try:
                shutil.copyfile(src, file_name)
            except shutil.SameFileError:
                pass  # Skip copy of downloaded files

        return (
            file_name + '.gz'
            if imported_format == ImportedFormat.COMPRESSED
            else file_name
        )

    # Large file download from Google Drive requires cookie and token.
    try:
        response = None
        if re.match(
            r'^https://drive.google.com/[-A-Za-z0-9\+&@#/%?=~_|!:,.;]*[-A-Za-z0-9\+&@#/%=~_|]$',
            src,
        ):
            session = requests.Session()
            response = session.get(src, stream=True)

            token = None
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    token = value
                    break

            if token is not None:
                params = {'confirm': token}
                response = session.get(src, params=params, stream=True)

        elif re.match(
            r'^(https?|ftp)://[-A-Za-z0-9\+&@#/%?=~_|!:,.;]*[-A-Za-z0-9\+&@#/%=~_|]$',
            src,
        ):
            response = requests.get(src, stream=True)
    except requests.exceptions.ConnectionError:
        raise requests.exceptions.ConnectionError("Could not connect to {}".format(src))

    if response:
        with open(file_name, 'wb') as f:
            total = response.headers.get('content-length')
            total = float(total) if total else None
            downloaded = 0
            current_progress = 0
            for content in response.iter_content(chunk_size=CHUNK_SIZE):
                f.write(content)

                if total is not None and progress_to is not None:
                    downloaded += len(content)
                    progress_span = progress_to - progress_from
                    next_progress = progress_from + progress_span * downloaded / total
                    next_progress = round(next_progress, 2)

                    if next_progress > current_progress:
                        print(progress(next_progress))
                        current_progress = next_progress

        # Check if a temporary file exists.
        if not os.path.isfile(file_name):
            raise ValueError("Downloaded file not found {}".format(file_name))

        src = file_name
    else:
        if not os.path.isfile(src):
            raise ValueError("Source file not found {}".format(src))

    # Decide which import should be used.
    if re.search(r'\.(bz2|zip|rar|7z|tgz|tar\.gz|tar\.bz2)$', file_name):
        destination_file_name = import7z()
    elif file_name.endswith('.gz'):
        destination_file_name = importGz()
    else:
        destination_file_name = importUncompressed()

    if progress_to is not None:
        print(progress(progress_to))

    return destination_file_name