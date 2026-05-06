def find_cache_directory(remote):
    """
    Find the directory where temporary local checkouts are to be stored.

    :returns: The absolute pathname of a directory (a string).
    """
    return os.path.join('/var/cache/vcs-repo-mgr' if os.access('/var/cache', os.W_OK) else tempfile.gettempdir(),
                        urlparse.quote(remote, safe=''))