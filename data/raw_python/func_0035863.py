def freeze(self, progressbar=None):
        """
        Convert :class:`dtoolcore.ProtoDataSet` to :class:`dtoolcore.DataSet`.
        """
        # Call the storage broker pre_freeze hook.
        self._storage_broker.pre_freeze_hook()

        if progressbar:
            progressbar.label = "Freezing dataset"

        # Generate and persist the manifest.
        manifest = self.generate_manifest(progressbar=progressbar)
        self._storage_broker.put_manifest(manifest)

        # Generate and persist overlays from any item metadata that has been
        # added.

        overlays = self._generate_overlays()
        for overlay_name, overlay in overlays.items():
            self._put_overlay(overlay_name, overlay)

        # Change the type of the dataset from "protodataset" to "dataset" and
        # add a "frozen_at" time stamp to the administrative metadata.
        datetime_obj = datetime.datetime.utcnow()
        metadata_update = {
            "type": "dataset",
            "frozen_at": dtoolcore.utils.timestamp(datetime_obj)
        }
        self._admin_metadata.update(metadata_update)
        self._storage_broker.put_admin_metadata(self._admin_metadata)

        # Clean up using the storage broker's post freeze hook.
        self._storage_broker.post_freeze_hook()