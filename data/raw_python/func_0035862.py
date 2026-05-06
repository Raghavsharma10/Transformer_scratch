def _generate_overlays(self):
        """Return dictionary of overlays generated from added item metadata."""
        overlays = defaultdict(dict)
        for handle in self._storage_broker.iter_item_handles():
            identifier = dtoolcore.utils.generate_identifier(handle)
            item_metadata = self._storage_broker.get_item_metadata(handle)
            for k, v in item_metadata.items():
                overlays[k][identifier] = v

        return overlays