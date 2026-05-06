def size(self, store_hashes=True):
        """
        Retrieves the size in bytes of this ZIP content.
        :return: Size of the zip content in bytes
        """
        if self.modified:
            self.__cache_content(store_hashes)

        return len(self.cached_content)