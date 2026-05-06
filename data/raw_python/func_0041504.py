def read_local_files(*file_paths: str) -> str:
    """
    Reads one or more text files and returns them joined together.
    A title is automatically created based on the file name.

    Args:
        *file_paths: list of files to aggregate

    Returns: content of files
    """

    def _read_single_file(file_path):
        with open(file_path) as f:
            filename = os.path.splitext(file_path)[0]
            title = f'{filename}\n{"=" * len(filename)}'
            return '\n\n'.join((title, f.read()))

    return '\n' + '\n\n'.join(map(_read_single_file, file_paths))