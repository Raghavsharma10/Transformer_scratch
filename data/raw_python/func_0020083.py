def name(self):
        '''
        Returns the name of the current :py:class:`Detrender` subclass.

        '''

        if self.cadence == 'lc':
            return self.__class__.__name__
        else:
            return '%s.sc' % self.__class__.__name__