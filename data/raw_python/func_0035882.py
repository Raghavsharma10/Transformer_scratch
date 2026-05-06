def roundedSpecClass(self):
        """ Spectral class with rounded class number ie A8.5V is A9 """
        try:
            classnumber = str(int(np.around(self.classNumber)))
        except TypeError:
            classnumber = str(self.classNumber)

        return self.classLetter + classnumber