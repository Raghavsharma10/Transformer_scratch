def recall(self, mode='all'):
        """Recalls from the file specified by :attr:`~SR850.filename`.

        :param mode: Specifies the recall mode.

            =======  ==================================================
            Value    Description
            =======  ==================================================
            'all'    Recalls the active display's data trace, the trace
                     definition and the instrument state.
            'state'  Recalls the instrument state.
            =======  ==================================================

        """
        if mode == 'all':
            self._write('RDAT')
        elif mode == 'state':
            self._write('RSET')
        else:
            raise ValueError('Invalid recall mode.')