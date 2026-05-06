def get_item_abspath(self, identifier):
        """Return absolute path at which item content can be accessed.

        :param identifier: item identifier
        :returns: absolute path from which the item content can be accessed
        """
        manifest = self.get_manifest()
        item = manifest["items"][identifier]
        item_abspath = os.path.join(self._data_abspath, item["relpath"])
        return item_abspath