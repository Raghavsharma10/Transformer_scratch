def _update_asset_content_filename_on_disk_to_match_id(self, ac):
        """Because we want the asset content filename to match the ac.ident,
        here we manipulate the saved file on disk after creating the
        asset content"""
        def has_secondary_storage():
            return 'secondary_data_store_path' in self._config_map

        datastore_path = ''
        secondary_data_store_path = ''

        if 'data_store_full_path' in self._config_map:
            datastore_path = self._config_map['data_store_full_path']
        if has_secondary_storage():
            secondary_data_store_path = self._config_map['secondary_data_store_path']
        relative_path = self._config_map['data_store_path']

        filepath = os.path.join(datastore_path, ac._my_map['url'])
        old_filename = os.path.splitext(os.path.basename(filepath))[0]
        new_path = filepath.replace(old_filename, ac.ident.identifier)
        os.rename(filepath, new_path)

        # Should also rename the file stored in the secondary path
        if has_secondary_storage():
            old_path = '{0}/repository/AssetContent'.format(relative_path)
            filepath = os.path.join(datastore_path, ac._my_map['url']).replace(old_path, secondary_data_store_path)
            old_filename = os.path.splitext(os.path.basename(filepath))[0]
            new_path = filepath.replace(old_filename, ac.ident.identifier)
            os.rename(filepath, new_path)