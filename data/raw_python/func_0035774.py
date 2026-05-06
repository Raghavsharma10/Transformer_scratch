def post_freeze_hook(self):
        """Post :meth:`dtoolcore.ProtoDataSet.freeze` cleanup actions.

        This method is called at the end of the
        :meth:`dtoolcore.ProtoDataSet.freeze` method.

        In the :class:`dtoolcore.storage_broker.DiskStorageBroker` it removes
        the temporary directory for storing item metadata fragment files.
        """
        if os.path.isdir(self._metadata_fragments_abspath):
            shutil.rmtree(self._metadata_fragments_abspath)