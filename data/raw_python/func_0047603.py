def set_data(self, data=None):
        """Sets the content data.

        arg:    data (osid.transport.DataInputStream): the content data
        raise:  InvalidArgument - ``data`` is invalid
        raise:  NoAccess - ``Metadata.isReadOnly()`` is ``true``
        raise:  NullArgument - ``data`` is ``null``
        *compliance: mandatory -- This method must be implemented.*

        """
        def has_secondary_storage():
            return 'secondary_data_store_path' in self._config_map

        extension = data.name.split('.')[-1]
        data_store_path = self._config_map['data_store_path']
        if has_secondary_storage():
            secondary_data_store_path = self._config_map['secondary_data_store_path']

        if '_id' in self._my_map:
            filename = self._my_map['_id']
            # remove any old file that is set
            if str(self._my_map['_id']) not in self._my_map['url']:
                os.remove(self._my_map['url'])

                if has_secondary_storage():
                    old_path = '{0}/repository/AssetContent'.format(data_store_path)
                    secondary_file_location = self._my_map['url'].replace(old_path,
                                                                          secondary_data_store_path)
                    os.remove(secondary_file_location)
        else:
            filename = ObjectId()

        filesystem_location = '{0}/repository/AssetContent/'.format(data_store_path)

        if not os.path.isdir(filesystem_location):
            os.makedirs(filesystem_location)

        file_location = '{0}{1}.{2}'.format(filesystem_location,
                                            str(filename),
                                            extension)

        data.seek(0)

        with open(file_location, 'wb') as file_handle:
            file_handle.write(data.read())

        # this URL should be a filesystem path...relative
        # to the setting in runtime
        self._payload.set_url(file_location)

        # if set, also make a backup copy in the secondary_data_store_path
        if has_secondary_storage():
            data.seek(0)

            if not os.path.isdir(secondary_data_store_path):
                os.makedirs(secondary_data_store_path)

            file_location = '{0}/{1}.{2}'.format(secondary_data_store_path,
                                                 str(filename),
                                                 extension)
            with open(file_location, 'wb') as file_handle:
                file_handle.write(data.read())