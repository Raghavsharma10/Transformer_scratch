def clearOldCalibrations(self, date=None):
        '''
        if not only a specific date than remove all except of the youngest calibration
        '''
        self.coeffs['dark current'] = [self.coeffs['dark current'][-1]]
        self.coeffs['noise'] = [self.coeffs['noise'][-1]]

        for light in self.coeffs['flat field']:
            self.coeffs['flat field'][light] = [
                self.coeffs['flat field'][light][-1]]
        for light in self.coeffs['lens']:
            self.coeffs['lens'][light] = [self.coeffs['lens'][light][-1]]