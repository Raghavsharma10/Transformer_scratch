def download(self, folder=None):
        # type: (Optional[str]) -> Tuple[str, str]
        """Download resource store to provided folder or temporary folder if no folder supplied

        Args:
            folder (Optional[str]): Folder to download resource to. Defaults to None.

        Returns:
            Tuple[str, str]: (URL downloaded, Path to downloaded file)

        """
        # Download the resource
        url = self.data.get('url', None)
        if not url:
            raise HDXError('No URL to download!')
        logger.debug('Downloading %s' % url)
        filename = self.data['name']
        format = '.%s' % self.data['format']
        if format not in filename:
            filename = '%s%s' % (filename, format)
        with Download(full_agent=self.configuration.get_user_agent()) as downloader:
            path = downloader.download_file(url, folder, filename)
            return url, path