def _get_asset_content(self, asset_id, asset_content_type_str=None, asset_content_id=None):
        """stub"""
        rm = self.my_osid_object._get_provider_manager('REPOSITORY')
        if 'assignedBankIds' in self.my_osid_object._my_map:
            if self.my_osid_object._proxy is not None:
                als = rm.get_asset_lookup_session_for_repository(
                    Id(self.my_osid_object._my_map['assignedBankIds'][0]),
                    self.my_osid_object._proxy)
            else:
                als = rm.get_asset_lookup_session_for_repository(
                    Id(self.my_osid_object._my_map['assignedBankIds'][0]))
        elif 'assignedBookIds' in self.my_osid_object._my_map:
            if self.my_osid_object._proxy is not None:
                als = rm.get_asset_lookup_session_for_repository(
                    Id(self.my_osid_object._my_map['assignedBookIds'][0]),
                    self.my_osid_object._proxy)
            else:
                als = rm.get_asset_lookup_session_for_repository(
                    Id(self.my_osid_object._my_map['assignedBookIds'][0]))
        elif 'assignedRepositoryIds' in self.my_osid_object._my_map:
            if self.my_osid_object._proxy is not None:
                als = rm.get_asset_lookup_session_for_repository(
                    Id(self.my_osid_object._my_map['assignedRepositoryIds'][0]),
                    self.my_osid_object._proxy)
            else:
                als = rm.get_asset_lookup_session_for_repository(
                    Id(self.my_osid_object._my_map['assignedRepositoryIds'][0]))
        else:
            raise KeyError

        if asset_content_id is not None:
            ac_list = als.get_asset(asset_id).get_asset_contents()
            for ac in ac_list:
                if str(ac.ident) == str(asset_content_id):
                    return ac

        if not asset_content_type_str:
            return next(als.get_asset(asset_id).get_asset_contents())  # Just return first one
        else:

            if isinstance(asset_content_type_str, Type):
                asset_content_type_str = str(asset_content_type_str)
            for ac in als.get_asset(asset_id).get_asset_contents():
                if ac.get_genus_type() == Type(asset_content_type_str):
                    return ac
        raise NotFound()