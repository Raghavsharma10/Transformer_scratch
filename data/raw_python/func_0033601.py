def detect(self, path):
        '''
        Guesses a TypeString from the given path
        '''
        typestring = UNKNOWN
        for detector in self.detectors:
            if typestring != UNKNOWN and not detector.can_improve(typestring):
                continue
            if not detector.can_detect(path):
                continue
            detected = detector.detect(path)
            if detected:
                typestring = detected
        return typestring