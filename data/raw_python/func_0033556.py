def strain_in_plane(self, **kwargs):
        '''
        Returns the in-plane strain assuming no lattice relaxation, which
        is positive for tensile strain and negative for compressive strain.
        '''
        if self._strain_out_of_plane is not None:
            return ((self._strain_out_of_plane / -2.) *
                    (self.unstrained.c11(**kwargs) /
                     self.unstrained.c12(**kwargs)  )  )
        else:
            return 1 - self.unstrained.a(**kwargs) / self.substrate.a(**kwargs)