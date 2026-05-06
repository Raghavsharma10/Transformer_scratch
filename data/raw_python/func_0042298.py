def from_name(cls, name, all_fallback=True):
        """Gets a vocation filter from a vocation's name.

        Parameters
        ----------
        name: :class:`str`
            The name of the vocation.
        all_fallback: :class:`bool`
            Whether to return :py:attr:`ALL` if no match is found. Otherwise, ``None`` will be returned.

        Returns
        -------
        VocationFilter, optional:
            The matching vocation filter.
        """
        name = name.upper()
        for vocation in cls:  # type: VocationFilter
            if vocation.name in name or vocation.name[:-1] in name and vocation != cls.ALL:
                return vocation
        if all_fallback or name.upper() == "ALL":
            return cls.ALL
        return None