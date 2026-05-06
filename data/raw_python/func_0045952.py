def clear_data(self):
        """Removes the content data.

        raise:  NoAccess - ``Metadata.isRequired()`` is ``true`` or
                ``Metadata.isReadOnly()`` is ``true``
        *compliance: mandatory -- This method must be implemented.*

        """
        if (self.get_data_metadata().is_read_only() or
                self.get_data_metadata().is_required()):
            raise errors.NoAccess()
        if self._my_map['data'] == self._data_default:
            return
        dbase = JSONClientValidated('repository',
                                    runtime=self._runtime).raw()
        filesys = gridfs.GridFS(dbase)
        filesys.delete(self._my_map['data'])
        self._my_map['data'] = self._data_default
        del self._my_map['base64']