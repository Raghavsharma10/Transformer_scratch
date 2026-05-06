def get_for_directory(
        dp,
        hash_mode="md5",
        filter_dots=False,
        filter_func=lambda fp:False
    ):
    r"""
        Returns a hash string for the files below a given directory path.

        :param dp:          Path to a directory.
        :param hash_mode:   Can be either one of 'md5', 'sha1', 'sha256' or 'sha512'.
                            Defines the algorithm used to generate the resulting hash
                            string. Default is 'md5'.
        :param filter_dots: If True will filter directories or files beginning with a '.' (dot) like '.git'.
                            Default is False.
        :param filter_func: A function receiving a path as a single paramter. If it returns True the given
                            path will be excluded from the hash calculation. Otherwise it will be included.
    """

    hash_func = _HASH_MODE_DICT.get(hash_mode)

    root_dps_fns =      os.walk( dp, topdown=True )
    root_dps_fns =      itertools.imap(         list,                  root_dps_fns )
    if filter_dots:
        root_dps_fns =  itertools.ifilterfalse( _is_dot_root,           root_dps_fns )
        root_dps_fns =  itertools.imap(         _filter_dot_fns,        root_dps_fns )
    fps_lists =         itertools.imap(         _gen_fps,               root_dps_fns )
    fps =               itertools.chain(        *fps_lists )
    fps =               itertools.ifilterfalse( filter_func,           fps )
    file_handles =      itertools.imap(         _get_file_handle,      fps )
    file_hash_digests = itertools.imap(         _get_file_hash_digest, file_handles, itertools.repeat(hash_func) )
    file_hash_digests = sorted( file_hash_digests )
    file_hash_digests = map(    _get_utf8_encoded, file_hash_digests )

    hash_ = _get_merged_hash( file_hash_digests, hash_func )

    return hash_.hexdigest()