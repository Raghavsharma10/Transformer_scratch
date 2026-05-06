def colors_json_ids(self):
        """ A proxy property based access to vertices in current edge

        When edge is serialized to JSON object, no explicit object for its multicolor is created, but rather all colors,
        taking into account their multiplicity, are referenced by their json_ids.
        """
        return [color.json_id if hasattr(color, "json_id") else hash(color) for color in self.multicolor.multicolors.elements()]