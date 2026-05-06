def write(self):
        """ write all needed state info to filesystem """
        dumped = self._fax.codec.dump(self.__state, open(self.state_file, 'w'))