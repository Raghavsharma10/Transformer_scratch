def primitive(self):
        """Primitive of the backbone.

        Notes
        -----
        This is the average of the positions of all the CAs in frames
        of `sl` `Residues`.
        """
        cas = self.get_reference_coords()
        primitive_coords = make_primitive_extrapolate_ends(
            cas, smoothing_level=self.sl)
        primitive = Primitive.from_coordinates(primitive_coords)
        primitive.relabel_monomers([x.id for x in self])
        primitive.id = self.id
        primitive.parent = self
        return primitive