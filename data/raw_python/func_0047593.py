def _get_catalog_idstrs(self):
        """Returns the proper list of catalog idstrs based on catalog view"""
        if self._catalog_view == ISOLATED:
            return [str(self._catalog_id)]
        else:
            return self._get_descendent_cat_idstrs(self._catalog_id)