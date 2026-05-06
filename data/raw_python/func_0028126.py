def get_location(self, locations=None):
        # type: (Optional[List[str]]) -> List[str]
        """Return the dataset's location

        Args:
            locations (Optional[List[str]]): Valid locations list. Defaults to list downloaded from HDX.

        Returns:
            List[str]: list of locations or [] if there are none
        """
        countries = self.data.get('groups', None)
        if not countries:
            return list()
        return [Locations.get_location_from_HDX_code(x['name'], locations=locations,
                                                     configuration=self.configuration) for x in countries]