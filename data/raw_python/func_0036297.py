def split_file(path, chunk_size=50000):
    """
    Splits the specified file into smaller files.
    """
    with open(path) as f:
        for i, lines in enumerate(_chunks(f, chunk_size)):
            file_split = '{}.{}'.format(os.path.basename(path), i)
            chunk_path = os.path.join('/tmp', file_split)
            with open(chunk_path, 'w') as f:
                f.writelines(lines)
            yield chunk_path