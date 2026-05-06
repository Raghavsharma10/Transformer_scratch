def set_orient(self):
        """ Return the computed orientation based on CD matrix. """
        self.orient = RADTODEG(N.arctan2(self.cd12,self.cd22))