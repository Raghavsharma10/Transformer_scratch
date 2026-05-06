def add_item_metadata(self, handle, key, value):
        """Store the given key:value pair for the item associated with handle.

        :param handle: handle for accessing an item before the dataset is
                       frozen
        :param key: metadata key
        :param value: metadata value
        """
        if not os.path.isdir(self._metadata_fragments_abspath):
            os.mkdir(self._metadata_fragments_abspath)

        prefix = self._handle_to_fragment_absprefixpath(handle)
        fpath = prefix + '.{}.json'.format(key)

        with open(fpath, 'w') as fh:
            json.dump(value, fh)