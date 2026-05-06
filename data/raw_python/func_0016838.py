def template_dict(self):
        """
        Returns a dictionary suitable for templating strings with
        properties and dimensions related to this Solver object.

        Used in templated GPU kernels.
        """
        slvr = self

        D = {
            # Constants
            'LIGHTSPEED': montblanc.constants.C,
        }

        # Map any types
        D.update(self.type_dict())

        # Update with dimensions
        D.update(self.dim_local_size_dict())

        # Add any registered properties to the dictionary
        for p in self._properties.itervalues():
            D[p.name] = getattr(self, p.name)

        return D