def delete_url(self, url):
        """
        Delete local files downloaded from given URL
        """
        # file may exist locally in compressed and decompressed states
        # delete both
        for decompress in [False, True]:
            key = (url, decompress)
            if key in self._local_paths:
                path = self._local_paths[key]
                remove(path)
                del self._local_paths[key]

            # possible that file was downloaded via the download module without
            # using the Cache object, this wouldn't end up in the local_paths
            # but should still be deleted
            path = self.local_path(
                url, decompress=decompress, download=False)

            if exists(path):
                remove(path)