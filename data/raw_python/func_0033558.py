def substrate_a(self, **kwargs):
        '''
        Returns the substrate's lattice parameter.
        '''
        if self.substrate is not None:
            return self.substrate.a(**kwargs)
        else:
            return (self.unstrained.a(**kwargs) /
                    (1. - self.strain_in_plane(**kwargs)))