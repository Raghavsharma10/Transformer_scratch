def clear_file(self):
        """stub"""
        if (self.get_file_metadata().is_read_only() or
                self.get_file_metadata().is_required()):
            raise NoAccess()
        if 'assetId' in self.my_osid_object_form._my_map['fileId']:
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

            aas.delete_asset(Id(self.my_osid_object_form._my_map['fileId']['assetId']))

        self.my_osid_object_form._my_map['fileId'] = \
            dict(self.get_file_metadata().get_default_object_values()[0])