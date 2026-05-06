def _update_asset_content_url_to_match_id(self, ac):
        """update the ac URL value to match the ident"""
        mgr = self._provider_session._get_provider_manager('REPOSITORY')
        aas = mgr.get_asset_admin_session_for_repository(self._provider_session._catalog_id,
                                                         proxy=self._provider_session._proxy)
        form = aas.get_asset_content_form_for_update(ac.ident)

        url = ac._my_map['url']
        old_url = os.path.splitext(os.path.basename(url))[0]
        new_url = url.replace(old_url, ac.ident.identifier)

        form.set_url(new_url)
        return aas.update_asset_content(form)