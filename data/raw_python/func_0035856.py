def _put_overlay(self, overlay_name, overlay):
        """Store overlay so that it is accessible by the given name.

        :param overlay_name: name of the overlay
        :param overlay: overlay must be a dictionary where the keys are
                        identifiers in the dataset
        :raises: TypeError if the overlay is not a dictionary,
                 ValueError if identifiers in overlay and dataset do not match
        """
        if not isinstance(overlay, dict):
            raise TypeError("Overlay must be dict")

        if set(self._identifiers()) != set(overlay.keys()):
            raise ValueError("Overlay keys must be dataset identifiers")

        self._storage_broker.put_overlay(overlay_name, overlay)