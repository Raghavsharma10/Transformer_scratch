def remove_location(self, location):
        # type: (str) -> bool
        """Remove a location. If the location is already added, it is ignored.

        Args:
            location (str): Location to remove

        Returns:
            bool: True if location removed or False if not
        """
        res = self._remove_hdxobject(self.data.get('groups'), location, matchon='name')
        if not res:
            res = self._remove_hdxobject(self.data.get('groups'), location.upper(), matchon='name')
        if not res:
            res = self._remove_hdxobject(self.data.get('groups'), location.lower(), matchon='name')
        return res