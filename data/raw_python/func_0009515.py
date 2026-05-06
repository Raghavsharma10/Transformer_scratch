def getCoeffStr(self):
        '''
        get the distortion coeffs in a formated string
        '''
        txt = ''
        for key, val in self.coeffs.items():
            txt += '%s = %s\n' % (key, val)
        return txt