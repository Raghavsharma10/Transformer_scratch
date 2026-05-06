def get(self, name, start, end, resolution, window):
        """
        :param name:
        :param start:
        :param end:
        :param resolution:
        :param window:
        :return: an analysed file.
        """
        logger.info(
            'Analysing ' + name + ' from ' + start + ' to ' + end + ' at ' + resolution + 'x resolution using ' + window + ' window')
        signal = self._uploadController.loadSignal(name,
                                                   start=start if start != 'start' else None,
                                                   end=end if end != 'end' else None)
        if signal is not None:
            window = tuple(filter(None, window.split(' ')))
            if len(window) == 2:
                window = (window[0], float(window[1]))
            import time
            data = {
                'spectrum': self._jsonify(
                    signal.spectrum(ref=SPECLAB_REFERENCE, segmentLengthMultiplier=int(resolution), window=window)
                ),
                'peakSpectrum': self._jsonify(
                    signal.peakSpectrum(ref=SPECLAB_REFERENCE, segmentLengthMultiplier=int(resolution), window=window)
                ),
                'analysedAt': int(time.time() * 1000)
            }
            return data, 200
        else:
            return None, 404