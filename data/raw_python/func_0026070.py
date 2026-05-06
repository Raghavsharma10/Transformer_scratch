def rotateCD(self,orient):
        """ Rotates WCS CD matrix to new orientation given by 'orient'
        """
        # Determine where member CRVAL position falls in ref frame
        # Find out whether this needs to be rotated to align with
        # reference frame.

        _delta = self.get_orient() - orient
        if _delta == 0.:
            return

        # Start by building the rotation matrix...
        _rot = fileutil.buildRotMatrix(_delta)
        # ...then, rotate the CD matrix and update the values...
        _cd = N.array([[self.cd11,self.cd12],[self.cd21,self.cd22]],dtype=N.float64)
        _cdrot = N.dot(_cd,_rot)
        self.cd11 = _cdrot[0][0]
        self.cd12 = _cdrot[0][1]
        self.cd21 = _cdrot[1][0]
        self.cd22 = _cdrot[1][1]
        self.orient = orient