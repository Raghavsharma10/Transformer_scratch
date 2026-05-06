def _add_asset_content(self,
                           asset_id,
                           asset_data=None,
                           asset_url=None,
                           asset_content_type=None,
                           asset_label=None):
        """stub"""
        rm = self.my_osid_object_form._get_provider_manager('REPOSITORY')
        try:
            # for create forms
            catalog_id = self.my_osid_object_form._catalog_id
        except AttributeError:
            # for update forms
            catalog_id = Id(self.my_osid_object_form._my_map['assignedBankIds'][0])

        try:
            aas = rm.get_asset_admin_session_for_repository(
                catalog_id,
                self.my_osid_object_form._proxy)
        except (TypeError, NullArgument):  # not a ProxyManager, so don't pass it the proxy
            aas = rm.get_asset_admin_session_for_repository(
                catalog_id)

        asset_content_type_list = []
        try:
            config = self.my_osid_object_form._runtime.get_configuration()
            parameter_id = Id('parameter:assetContentRecordTypeForFiles@json')
            asset_content_type_list.append(
                config.get_value_by_parameter(parameter_id).get_type_value())
        except (AttributeError, KeyError):
            pass

        acfc = aas.get_asset_content_form_for_create(asset_id,
                                                     asset_content_type_list)
        if asset_content_type is not None:
            acfc.set_genus_type(asset_content_type)
        if asset_label is not None:
            acfc.display_name = str(asset_label)
        if asset_data:
            acfc.set_data(asset_data)
        if asset_url:
            acfc.set_url(asset_url)
        ac = aas.create_asset_content(acfc)

        return asset_id, ac.ident