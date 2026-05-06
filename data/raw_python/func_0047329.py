def _get_catalog_hierarchy_id(self, catalog_id, proxy, runtime):
        """Gets the catalog hierarchy"""
        seed_str = convert_catalog_id_to_object_id_string(catalog_id)
        ident = Id(authority=self._authority,
                   namespace='hierarchy.Hierarchy',
                   identifier=seed_str)
        return HierarchyLookupSession(proxy, runtime).get_hierarchy(ident).get_id()