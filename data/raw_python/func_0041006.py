def dest_fpath(self, source_fpath: str) -> str:
        """Calculates full path for end json-api file from source file full
        path."""
        relative_fpath = os.path.join(*source_fpath.split(os.sep)[1:])
        relative_dirpath = os.path.dirname(relative_fpath)

        source_fname = relative_fpath.split(os.sep)[-1]
        base_fname = source_fname.split('.')[0]
        dest_fname = f'{base_fname}.json'

        return os.path.join(self.dest_dir, relative_dirpath, dest_fname)