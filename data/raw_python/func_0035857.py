def generate_manifest(self, progressbar=None):
        """Return manifest generated from knowledge about contents."""
        items = dict()

        if progressbar:
            progressbar.label = "Generating manifest"

        for handle in self._storage_broker.iter_item_handles():
            key = dtoolcore.utils.generate_identifier(handle)
            value = self._storage_broker.item_properties(handle)
            items[key] = value
            if progressbar:
                progressbar.item_show_func = lambda x: handle
                progressbar.update(1)

        manifest = {
            "items": items,
            "dtoolcore_version": __version__,
            "hash_function": self._storage_broker.hasher.name
        }

        return manifest