def _get_phantom_root_catalog(self, cat_name, cat_class):
        """Get's the catalog id corresponding to the root of all implementation catalogs."""
        catalog_map = make_catalog_map(cat_name, identifier=PHANTOM_ROOT_IDENTIFIER)
        return cat_class(osid_object_map=catalog_map, runtime=self._runtime, proxy=self._proxy)