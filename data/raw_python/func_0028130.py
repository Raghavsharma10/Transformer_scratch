def add_other_location(self, location, exact=True, alterror=None, locations=None):
        # type: (str, bool, Optional[str], Optional[List[str]]) -> bool
        """Add a location which is not a country or region. Value is parsed and compared to existing locations in
        HDX. If the location is already added, it is ignored.

        Args:
            location (str): Location to add
            exact (bool): True for exact matching or False to allow fuzzy matching. Defaults to True.
            alterror (Optional[str]): Alternative error message to builtin if location not found. Defaults to None.
            locations (Optional[List[str]]): Valid locations list. Defaults to list downloaded from HDX.

        Returns:
            bool: True if location added or False if location already present
        """
        hdx_code, match = Locations.get_HDX_code_from_location_partial(location, locations=locations,
                                                                       configuration=self.configuration)
        if hdx_code is None or (exact is True and match is False):
            if alterror is None:
                raise HDXError('Location: %s - cannot find in HDX!' % location)
            else:
                raise HDXError(alterror)
        groups = self.data.get('groups', None)
        hdx_code = hdx_code.lower()
        if groups:
            if hdx_code in [x['name'] for x in groups]:
                return False
        else:
            groups = list()
        groups.append({'name': hdx_code})
        self.data['groups'] = groups
        return True