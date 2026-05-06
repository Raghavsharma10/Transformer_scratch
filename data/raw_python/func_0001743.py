def download_file(self, url, folder=None, filename=None, overwrite=False,
                      post=False, parameters=None, timeout=None):
        # type: (str, Optional[str], Optional[str], bool, bool, Optional[Dict], Optional[float]) -> str
        """Download file from url and store in provided folder or temporary folder if no folder supplied

        Args:
            url (str): URL to download
            folder (Optional[str]): Folder to download it to. Defaults to None.
            filename (Optional[str]): Filename to use for downloaded file. Defaults to None (derive from the url).
            overwrite (bool): Whether to overwrite existing file. Defaults to False.
            post (bool): Whether to use POST instead of GET. Defaults to False.
            parameters (Optional[Dict]): Parameters to pass. Defaults to None.
            timeout (Optional[float]): Timeout for connecting to URL. Defaults to None (no timeout).

        Returns:
            str: Path of downloaded file

        """
        self.setup(url, stream=True, post=post, parameters=parameters, timeout=timeout)
        return self.stream_file(url, folder, filename, overwrite)