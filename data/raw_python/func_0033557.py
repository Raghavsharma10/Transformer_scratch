def strain_out_of_plane(self, **kwargs):
        '''
        Returns the out-of-plane strain assuming no lattice relaxation, which
        is negative for tensile strain and positive for compressive strain.
        This is the strain measured by X-ray diffraction (XRD) symmetric
        omega-2theta scans.
        '''
        if self._strain_out_of_plane is not None:
            return self._strain_out_of_plane
        else:
            return (-2 * self.unstrained.c12(**kwargs) /
                    self.unstrained.c11(**kwargs) *
                    self.strain_in_plane(**kwargs)      )