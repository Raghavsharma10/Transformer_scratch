def hash_stream(self, url):
        # type: (str) -> str
        """Stream file from url and hash it using MD5. Must call setup method first.

        Args:
            url (str): URL to download

        Returns:
            str: MD5 hash of file

        """
        md5hash = hashlib.md5()
        try:
            for chunk in self.response.iter_content(chunk_size=10240):
                if chunk:  # filter out keep-alive new chunks
                    md5hash.update(chunk)
            return md5hash.hexdigest()
        except Exception as e:
            raisefrom(DownloadError, 'Download of %s failed in retrieval of stream!' % url, e)