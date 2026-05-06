def get_data(self):
        """Gets the asset content data.

        return: (osid.transport.DataInputStream) - the length of the
                content data
        raise:  OperationFailed - unable to complete request
        *compliance: mandatory -- This method must be implemented.*

        """
        if not bool(self._my_map['data']):
            raise errors.IllegalState('no data')
        dbase = JSONClientValidated('repository',
                                    runtime=self._runtime).raw()
        filesys = gridfs.GridFS(dbase)
        return DataInputStream(filesys.get(self._my_map['data']))