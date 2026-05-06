def get_for_file( fp, hash_mode="md5" ):
    r"""
        Returns a hash string for the given file path.

        :param fp:          Path to the file.
        :param hash_mode:   Can be either one of 'md5', 'sha1', 'sha256' or 'sha512'.
                            Defines the algorithm used to generate the resulting hash
                            string. Default is 'md5'.
    """

    with _get_file_handle(fp) as f:
        file_hash_digest = get_for_handle(f, hash_mode)

        return file_hash_digest