def remove_die(self, die):
        '''Remove ``Die`` (first matching) from Roll.
        :param die: Die instance
        '''
        if die in self._dice:
            self._dice.remove(die)