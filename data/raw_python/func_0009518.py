def readFromFile(self, filename):
        '''
        read the distortion coeffs from file
        '''
        s = dict(np.load(filename))
        try:
            self.coeffs = s['coeffs'][()]
        except KeyError:
            #LEGENCY - remove
            self.coeffs = s
        try:
            self.opts = s['opts'][()]
        except KeyError:
            pass
        return self.coeffs