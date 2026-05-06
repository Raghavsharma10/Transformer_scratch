def overview(self):
        '''
        Returns:
            str: an overview covering all calibrations 
            infos and shapes
        '''
        c = self.coeffs
        out = 'camera name: %s' % c['name']
        out += '\nmax value: %s' % c['depth']
        out += '\nlight spectra: %s' % c['light spectra']

        out += '\ndark current:'
        for (date, info, (slope, intercept), error) in c['dark current']:
            out += '\n\t date: %s' % self._toDateStr(date)
            out += '\n\t\t info: %s; slope:%s, intercept:%s' % (
                info, slope.shape, intercept.shape)

        out += '\nflat field:'
        for light, vals in c['flat field'].items():
            out += '\n\t light: %s' % light
            for (date, info, arr, error) in vals:
                out += '\n\t\t date: %s' % self._toDateStr(date)
                out += '\n\t\t\t info: %s; array:%s' % (info, arr.shape)

        out += '\nlens:'
        for light, vals in c['lens'].items():
            out += '\n\t light: %s' % light
            for (date, info, coeffs) in vals:
                out += '\n\t\t date: %s' % self._toDateStr(date)
                out += '\n\t\t\t info: %s; coeffs:%s' % (info, coeffs)

        out += '\nnoise:'
        for (date, info, nlf_coeff, error) in c['noise']:
            out += '\n\t date: %s' % self._toDateStr(date)
            out += '\n\t\t info: %s; coeffs:%s' % (info, nlf_coeff)

        out += '\nPoint spread function:'
        for light, vals in c['psf'].items():
            out += '\n\t light: %s' % light
            for (date, info, psf) in vals:
                out += '\n\t\t date: %s' % self._toDateStr(date)
                out += '\n\t\t\t info: %s; shape:%s' % (info, psf.shape)

        return out