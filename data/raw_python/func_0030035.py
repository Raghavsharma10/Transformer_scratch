def partition(self):
        """For partition urltypes, return the partition specified by the ref """

        if self.urltype != 'partition':
            return None

        return self._bundle.library.partition(self.url)