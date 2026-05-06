def get_report_hook(self):
        """
        Return a callback function suitable for using reporthook argument of urllib(.request).urlretrieve
        :return: function object
        """
        def report_hook(chunkNumber, chunkSize, totalSize):
            if totalSize != -1 and not self._callback.range_initialized():
                log.debug('Initializing range: [{},{}]'.format(0, totalSize))
                self._callback.set_range(0, totalSize)
            self._chunkNumber = chunkNumber
            self._total += chunkSize
            if self._total > totalSize:
                # The chunk size can be bigger than the file
                self._total = totalSize
            self._callback.update(self._total)

        return report_hook