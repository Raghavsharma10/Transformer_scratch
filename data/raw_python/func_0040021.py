def phys_name(self, item: str) -> str:
        """Return the physical (mapped) name of item.

        :param item: logical table name
        :return: physical name of table
        """
        v = self.__dict__[item]
        return v if v is not None else item