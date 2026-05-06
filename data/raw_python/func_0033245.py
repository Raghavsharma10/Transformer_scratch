def strained_001(self, target):
        '''
        Returns an instance of ``IIIVZincBlendeStrained001``, which is a
        biaxial-strained III-V zinc blende binary alloy grown on a (001)
        surface.
        
        Parameters
        ----------
        target : Alloy with ``a`` parameter or float
            Growth substrate, assumed to have a (001) surface, or out-of-plane
            strain, which is negative for tensile strain and positive for
            compressive strain. This is the strain measured by X-ray
            diffraction (XRD) symmetric omega-2theta scans.
        '''
        if isinstance(target, Alloy):
            return IIIVZincBlendeStrained001(unstrained=self,
                                             substrate=target)
        else:
            return IIIVZincBlendeStrained001(unstrained=self,
                                             strain_out_of_plane=target)