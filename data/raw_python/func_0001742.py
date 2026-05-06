def stream_file(self, url, folder=None, filename=None, overwrite=False):
        # type: (str, Optional[str], Optional[str], bool) -> str
        """Stream file from url and store in provided folder or temporary folder if no folder supplied.
        Must call setup method first.

        Args:
            url (str): URL to download
            filename (Optional[str]): Filename to use for downloaded file. Defaults to None (derive from the url).
            folder (Optional[str]): Folder to download it to. Defaults to None (temporary folder).
            overwrite (bool): Whether to overwrite existing file. Defaults to False.

        Returns:
            str: Path of downloaded file

        """
        path = self.get_path_for_url(url, folder, filename, overwrite)
        f = None
        try:
            f = open(path, 'wb')
            for chunk in self.response.iter_content(chunk_size=10240):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    f.flush()
            return f.name
        except Exception as e:
            raisefrom(DownloadError, 'Download of %s failed in retrieval of stream!' % url, e)
        finally:
            if f:
                f.close()