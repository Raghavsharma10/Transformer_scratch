def all_tables(self) -> List[str]:
        """
        List of all known tables
        :return:
        """
        return sorted([k for k in self.__dict__.keys()
                       if k not in _I2B2Tables._funcs and not k.startswith("_")])