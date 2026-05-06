def _view_filter(self):
        """
        Returns the mongodb catalog filter for isolated or federated views.

        This also searches across all underlying catalogs in federated
        catalog views. Real authz for controlling access to underlying
        catalogs will need to be managed in an adapter above the
        pay grade of this implementation.

        """
        if self._is_phantom_root_federated():
            return {}
        idstr_list = self._get_catalog_idstrs()
        return {'assigned' + self._catalog_name + 'Ids': {'$in': idstr_list}}