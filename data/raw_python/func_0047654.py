def clear_preview(self):
        """stub"""
        try:
            rm = self.my_osid_object._get_provider_manager('REPOSITORY')
        except AttributeError:
            rm = self.my_osid_object_form._get_provider_manager('REPOSITORY')
        try:
            aas = rm.get_asset_admin_session_for_repository(
                Id(self.my_osid_object._my_map['assignedBankIds'][0]))
        except AttributeError:
            # for update forms
            aas = rm.get_asset_admin_session_for_repository(
                Id(self.my_osid_object_form._my_map['assignedBankIds'][0]))
        if 'preview' not in self.my_osid_object_form._my_map['fileIds']:
            raise NotFound()
        aas.delete_asset(
            Id(self.my_osid_object_form._my_map['fileIds']['preview']['assetId']))
        del self.my_osid_object_form._my_map['fileIds']['preview']