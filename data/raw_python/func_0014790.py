def FilePattern(pattern, settings, **kwargs):
    """Factory method returns LocalFilePattern or GoogleStorageFilePattern
    """
    url = _urlparse(pattern)
    if url.scheme == 'gs':
        return GoogleStorageFilePattern(pattern, settings, **kwargs)
    else:
        assert url.scheme == 'file'
        return LocalFilePattern(pattern, settings, **kwargs)