def clear_files(self):
        """stub"""
        # This could also be implemented by iterating over self.clear_file()
        if self._files_metadata['required'] or self._files_metadata['read_only']:
            raise NoAccess()
        rm = self.my_osid_object_form._get_provider_manager('REPOSITORY')

        catalog_id_str = ''

        if 'assignedBankIds' in self.my_osid_object_form._my_map:
            catalog_id_str = self.my_osid_object_form._my_map['assignedBankIds'][0]
        elif 'assignedRepositoryIds' in self.my_osid_object_form._my_map:
            catalog_id_str = self.my_osid_object_form._my_map['assignedRepositoryIds'][0]
        try:
            try:
                aas = rm.get_asset_admin_session_for_repository(
                    Id(catalog_id_str),
                    self.my_osid_object_form._proxy)
            except NullArgument:
                aas = rm.get_asset_admin_session_for_repository(
                    Id(catalog_id_str))
        except AttributeError:
            # for update forms
            try:
                aas = rm.get_asset_admin_session_for_repository(
                    Id(catalog_id_str),
                    self.my_osid_object_form._proxy)
            except NullArgument:
                aas = rm.get_asset_admin_session_for_repository(
                    Id(catalog_id_str))
        for label, asset_data in self.my_osid_object_form._my_map['fileIds'].items():
            aas.delete_asset(Id(asset_data['assetId']))
        self.my_osid_object_form._my_map['fileIds'] = \
            dict(self._files_metadata['default_object_values'][0])