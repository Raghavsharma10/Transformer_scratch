def register(self, switch, signal=signals.switch_registered):
        '''
        Register a switch and persist it to the storage.
        '''
        if not switch.name:
            raise ValueError('Switch name cannot be blank')

        switch.manager = self
        self.__persist(switch)

        signal.call(switch)