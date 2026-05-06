def swap(self):
        # type: () -> DictWrapper
        """Swap key and value

        /!\ Be carreful, if there are duplicate values, only one will
        survive /!\

        Example:

            >>> from ww import d
            >>> d({1: 2, 2: 2, 3: 3}).swap()
            {2: 2, 3: 3}
        """
        return self.__class__((v, k) for k, v in self.items())