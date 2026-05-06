def import_from_xml(self, xml):
        '''
        Standard imports for all types of object
        These must fail gracefully, skip if not found
        '''
        self._import_orgid(xml)
        self._import_parents_from_xml(xml)
        self._import_instances_from_xml(xml)
        self._import_common_name(xml)
        self._import_synonyms(xml)
        self._import_dblinks(xml)