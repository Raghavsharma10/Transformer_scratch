def update_name(self, new_name):
        """Update the name of the proto dataset.

        :param new_name: the new name of the proto dataset
        """

        if not dtoolcore.utils.name_is_valid(new_name):
            raise(DtoolCoreInvalidNameError())

        self._admin_metadata['name'] = new_name
        if self._storage_broker.has_admin_metadata():
            self._storage_broker.put_admin_metadata(self._admin_metadata)