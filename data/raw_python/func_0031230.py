def subcellular_locations(self):
        """Distinct subcellular locations (``location`` in :class:`.models.SubcellularLocation`)

        :return: all distinct subcellular locations
        :rtype: list[str]
        """
        return [x[0] for x in self.session.query(models.SubcellularLocation.location).all()]