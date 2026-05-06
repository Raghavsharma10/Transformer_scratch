def get_item_metadata(self, handle):
        """Return dictionary containing all metadata associated with handle.

        In other words all the metadata added using the ``add_item_metadata``
        method.

        :param handle: handle for accessing an item before the dataset is
                       frozen
        :returns: dictionary containing item metadata
        """

        if not os.path.isdir(self._metadata_fragments_abspath):
            return {}

        prefix = self._handle_to_fragment_absprefixpath(handle)

        def list_abspaths(dirname):
            for f in os.listdir(dirname):
                yield os.path.join(dirname, f)

        files = [f for f in list_abspaths(self._metadata_fragments_abspath)
                 if f.startswith(prefix)]

        metadata = {}
        for f in files:
            key = f.split('.')[-2]  # filename: identifier.key.json
            with open(f) as fh:
                value = json.load(fh)
            metadata[key] = value

        return metadata