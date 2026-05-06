def _set_asset(self,
                   asset_data=None,
                   asset_type=None,
                   asset_content_type=None,
                   asset_content_record_types=None,
                   display_name='',
                   description=''):
        """stub"""
        # This method should be deprecated and its code added to the create_asset method:
        rm = self.my_osid_object_form._get_provider_manager('REPOSITORY')
        catalog_id = ''
        try:
            # for create forms
            catalog_id = self.my_osid_object_form._catalog_id
        except AttributeError:
            # for update forms
            if 'assignedBankIds' in self.my_osid_object_form._my_map:
                catalog_id = Id(self.my_osid_object_form._my_map['assignedBankIds'][0])
            elif 'assignedRepositoryIds' in self.my_osid_object_form._my_map:
                catalog_id = Id(self.my_osid_object_form._my_map['assignedRepositoryIds'][0])

        try:
            aas = rm.get_asset_admin_session_for_repository(
                catalog_id,
                self.my_osid_object_form._proxy)
        except (TypeError, NullArgument):  # not a ProxyManager, so don't pass it the proxy
            aas = rm.get_asset_admin_session_for_repository(
                catalog_id)
        afc = aas.get_asset_form_for_create([])
        if asset_type is not None:
            afc.set_genus_type(asset_type)

        afc.set_display_name(display_name)
        afc.set_description(description)
        asset_id = aas.create_asset(afc).get_id()
        ac = None
        if asset_data is not None:
            asset_content_type_list = asset_content_record_types
            if asset_content_type_list is None:
                asset_content_type_list = []

            try:
                config = self.my_osid_object_form._runtime.get_configuration()
                parameter_id = Id('parameter:assetContentRecordTypeForFiles@json')
                asset_content_type_list.append(
                    config.get_value_by_parameter(parameter_id).get_type_value())
            except (AttributeError, KeyError, NotFound):
                pass

            acfc = aas.get_asset_content_form_for_create(asset_id,
                                                         asset_content_type_list)
            if asset_content_type is not None:
                acfc.set_genus_type(asset_content_type)

            acfc.set_data(asset_data)
            ac = aas.create_asset_content(acfc)

            # really stupid, but set the data again, because for filesystem impl
            # the ID above will be off by one-ish -- we need it to match the
            # AssetContent ID, so re-set it.
            # have to set it above so that the filesystem adapter kicks in on update
            # asset_data.seek(0)
            # acfu = aas.get_asset_content_form_for_update(ac.ident)
            # acfu.set_data(asset_data)
            # ac = aas.update_asset_content(acfu)
        if ac is not None:
            return asset_id, ac.ident
        else:
            return asset_id, None