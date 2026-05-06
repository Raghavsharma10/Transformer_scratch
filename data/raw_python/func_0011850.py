def check_definition(self):
        """
        called after Defintion was loaded to sanity check
        raises on error
        """
        if not self.write_codec:
            self.__write_codec = self.defined.data_ext

        # TODO need to add back a class scope target limited for subprojects with sub target sets
        targets = self.get_defined_targets()
        if self.__target_only:
            if self.__target_only not in targets:
                raise RuntimeError("invalid target '%s'" % self.__target_only)

            self.targets = [self.__target_only]
        else:
            self.targets = targets