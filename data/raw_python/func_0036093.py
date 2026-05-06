def _getParLabelAndUnit(self, param):
        """ checks param to see if it contains a parent link (ie star.) then returns the correct unit and label for the
         job from the parDicts
        :return:
        """

        firstObject = self.objectList[0]

        if isinstance(firstObject, ac.Planet):
            if 'star.' in param:
                return _starPars[param[5:]]  # cut off star. part
            else:
                return _planetPars[param]
        elif isinstance(firstObject, ac.Star):
            return _starPars[param]
        else:
            raise TypeError('Only Planets and Star object are currently supported, you gave {0}'.format(type(firstObject)))