def clear_file(self, label):
        """stub"""
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

        if label not in self.my_osid_object_form._my_map['fileIds']:
            raise NotFound()
        aas.delete_asset(Id(self.my_osid_object_form._my_map['fileIds'][label]['assetId']))
        del self.my_osid_object_form._my_map['fileIds'][label]