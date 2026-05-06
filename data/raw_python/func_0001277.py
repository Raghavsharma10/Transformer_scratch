def add_potential(self, *patterns):
        ''' Add a potential config file pattern '''
        for ptn in patterns:
            self.__potential.extend(self._ptn2fn(ptn))