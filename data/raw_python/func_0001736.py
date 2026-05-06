def get_path_for_url(url, folder=None, filename=None, overwrite=False):
        # type: (str, Optional[str], Optional[str], bool) -> str
        """Get filename from url and join to provided folder or temporary folder if no folder supplied, ensuring uniqueness

        Args:
            url (str): URL to download
            folder (Optional[str]): Folder to download it to. Defaults to None (temporary folder).
            filename (Optional[str]): Filename to use for downloaded file. Defaults to None (derive from the url).
            overwrite (bool): Whether to overwrite existing file. Defaults to False.

        Returns:
            str: Path of downloaded file

        """
        if not filename:
            urlpath = urlsplit(url).path
            filename = basename(urlpath)
        filename, extension = splitext(filename)
        if not folder:
            folder = get_temp_dir()
        path = join(folder, '%s%s' % (filename, extension))
        if overwrite:
            try:
                remove(path)
            except OSError:
                pass
        else:
            count = 0
            while exists(path):
                count += 1
                path = join(folder, '%s%d%s' % (filename, count, extension))
        return path