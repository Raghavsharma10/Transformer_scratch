def zSetSurfaceData(self, surfNum, radius=None, thick=None, material=None, semidia=None, 
                        conic=None, comment=None):
        """Sets surface data"""
        if self.pMode == 0: # Sequential mode
            surf = self.pLDE.GetSurfaceAt(surfNum)
            if radius is not None:
                surf.pRadius = radius
            if thick is not None:
                surf.pThickness = thick
            if material is not None:
                surf.pMaterial = material
            if semidia is not None:
                surf.pSemiDiameter = semidia
            if conic is not None:
                surf.pConic = conic
            if comment is not None:
                surf.pComment = comment
        else:
            raise NotImplementedError('Function not implemented for non-sequential mode')