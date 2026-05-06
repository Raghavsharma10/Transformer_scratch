def _get_file(self, fn, timeout=30):
        '''Request a SiriDB configuration file.

        This method is used by the SiriDB manage tool and should not be used
        otherwise. Full access rights are required for this request.
        '''
        msg = FILE_MAP.get(fn, None)
        if msg is None:
            raise FileNotFoundError('Cannot get file {!r}. Available file '
                                    'requests are: {}'
                                    .format(fn, ', '.join(FILE_MAP.keys())))
        result = self._loop.run_until_complete(
            self._protocol.send_package(msg, timeout=timeout))
        return result