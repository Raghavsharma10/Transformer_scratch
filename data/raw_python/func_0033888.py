def get_for_handle( f, hash_mode="md5" ):
    r"""
        Returns a hash string for the given file-like object.

        :param f:           The file object.
        :param hash_mode:   Can be either one of 'md5', 'sha1', 'sha256' or 'sha512'.
                            Defines the algorithm used to generate the resulting hash
                            string. Default is 'md5'.
    """

    hash_func = _HASH_MODE_DICT.get(hash_mode)
    file_hash_digest = _get_file_hash_digest( f, hash_func )
    file_hash_digest = _get_utf8_encoded( file_hash_digest )
    return file_hash_digest