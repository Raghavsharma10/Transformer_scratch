def writeToFile(self, filename, saveOpts=False):
        '''
        write the distortion coeffs to file
        saveOpts --> Whether so save calibration options (and not just results)
        '''
        try:
            if not filename.endswith('.%s' % self.ftype):
                filename += '.%s' % self.ftype
            s = {'coeffs': self.coeffs}
            if saveOpts:
                s['opts'] = self.opts
#             else:
#                 s['opts':{}]
            np.savez(filename, **s)
            return filename
        except AttributeError:
            raise Exception(
                'need to calibrate camera before calibration can be saved to file')