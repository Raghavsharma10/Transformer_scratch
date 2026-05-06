def analyse(self, name):
        """
        reads the specified file.
        :param name: the name.
        :return: the analysis as frequency/Pxx.
        """
        if name in self._cache:
            target = self._cache[name]
            if target['type'] == 'wav':
                signal = self._uploadController.loadSignal(target['filename'],
                                                           start=target['start'] if target['start'] != 'start' else None,
                                                           end=target['end'] if target['end'] != 'end' else None)
                if signal is not None:
                    # TODO allow user defined window
                    return getattr(signal, target['analysis'])(ref=1.0)
                else:
                    return None, 404
                pass
            elif target['type'] == 'hinge':
                hingePoints = np.array(target['hinge']).astype(np.float64)
                x = hingePoints[:, 1]
                y = hingePoints[:, 0]
                # extend as straight line from 0 to 500
                if x[0] != 0:
                    x = np.insert(x, 0, 0.0000001)
                    y = np.insert(y, 0, y[0])
                if x[-1] != 500:
                    x = np.insert(x, len(x), 500.0)
                    y = np.insert(y, len(y), y[-1])
                # convert the y axis dB values into a linear value
                y = 10 ** (y / 10)
                # perform a logspace interpolation
                f = self.log_interp1d(x, y)
                # remap to 0-500
                xnew = np.linspace(x[0], x[-1], num=500, endpoint=False)
                # and convert back to dB
                return xnew, 10 * np.log10(f(xnew))
            else:
                logger.error('Unknown target type with name ' + name)
        return None