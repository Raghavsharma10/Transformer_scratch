def _get_signal_event(self, s):
        '''
        Get the event for a signal.

        Checks if the signal has been enabled and raises a
        ``ValueError`` if not.
        '''
        try:
            return self._signal_events[int(s)]
        except KeyError:
            raise ValueError('Signal {} has not been enabled'.format(s))