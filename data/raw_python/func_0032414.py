def save(self, mode='all'):
        """Saves to the file specified by :attr:`~SR850.filename`.

        :param mode: Defines what to save.

            =======  ================================================
            Value    Description
            =======  ================================================
            'all'    Saves the active display's data trace, the trace
                     definition and the instrument state.
            'data'   Saves the active display's data trace.
            'state'  Saves the instrument state.
            =======  ================================================

        """
        if mode == 'all':
            self._write('SDAT')
        elif mode == 'data':
            self._write('SASC')
        elif mode=='state':
            self._write('SSET')
        else:
            raise ValueError('Invalid save mode.')