def _normalize_file_path(file_path):
        """Normalize the file path value.

        :param str file_path: The file path as passed in
        :rtype: str

        """
        if not file_path:
            return None
        elif file_path.startswith('s3://') or \
                file_path.startswith('http://') or \
                file_path.startswith('https://'):
            return file_path
        return path.abspath(file_path)