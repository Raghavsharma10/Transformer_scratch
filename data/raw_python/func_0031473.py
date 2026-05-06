def get_subcellular_locations(self, entry):
        """
        get list of models.SubcellularLocation object from XML node entry

        :param entry: XML node entry
        :return: list of :class:`pyuniprot.manager.models.SubcellularLocation` object
        """
        subcellular_locations = []
        query = './comment/subcellularLocation/location'
        sls = {x.text for x in entry.iterfind(query)}

        for sl in sls:

            if sl not in self.subcellular_locations:
                self.subcellular_locations[sl] = models.SubcellularLocation(location=sl)
            subcellular_locations.append(self.subcellular_locations[sl])

        return subcellular_locations