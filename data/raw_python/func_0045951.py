def set_data(self, data):
        """Sets the content data.

        arg:    data (osid.transport.DataInputStream): the content data
        raise:  InvalidArgument - ``data`` is invalid
        raise:  NoAccess - ``Metadata.isReadOnly()`` is ``true``
        raise:  NullArgument - ``data`` is ``null``
        *compliance: mandatory -- This method must be implemented.*

        """
        if data is None:
            raise errors.NullArgument('data cannot be None')
        if not isinstance(data, DataInputStream):
            raise errors.InvalidArgument('data must be instance of DataInputStream')
        dbase = JSONClientValidated('repository',
                                    runtime=self._runtime).raw()
        filesys = gridfs.GridFS(dbase)
        self._my_map['data'] = filesys.put(data._my_data)
        data._my_data.seek(0)
        self._my_map['base64'] = base64.b64encode(data._my_data.read())