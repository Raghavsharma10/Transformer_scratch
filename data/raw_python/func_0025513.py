def is_matching(cls, file_path):
        """
            Return whether the given absolute file path is an ndata file.
        """
        if file_path.endswith(".ndata") and os.path.exists(file_path):
            try:
                with open(file_path, "r+b") as fp:
                    local_files, dir_files, eocd = parse_zip(fp)
                    contains_data = b"data.npy" in dir_files
                    contains_metadata = b"metadata.json" in dir_files
                    file_count = contains_data + contains_metadata  # use fact that True is 1, False is 0
                    # TODO: make sure ndata isn't compressed, or handle it
                    if len(dir_files) != file_count or file_count == 0:
                        return False
                    return True
            except Exception as e:
                logging.error("Exception parsing ndata file: %s", file_path)
                logging.error(str(e))
        return False