def _create_catalog_hierarchy(self, catalog_id, proxy, runtime):
        """Creates a catalog hierarchy"""
        seed_str = convert_catalog_id_to_object_id_string(catalog_id)
        has = HierarchyAdminSession(proxy, runtime)
        hfc = has.get_hierarchy_form_for_create([])
        hfc.set_display_name(catalog_id.get_identifier().title() + ' Hierarchy')
        hfc.set_description(
            'Hierarchy for ' + catalog_id.get_authority().title() +
            ' ' + catalog_id.get_identifier().title())
        hfc.set_genus_type(Type(authority='DLKIT',
                                namespace='hierarchy.Hierarchy',
                                identifier=catalog_id.get_identifier().lower()))
        # This next tricks require serious inside knowledge:
        hfc._my_map['_id'] = ObjectId(seed_str)
        hierarchy = has.create_hierarchy(hfc)
        return hierarchy.get_id()