def create(cls, name, members):
        """Creates a new enum type based on this one (cls) and adds newly
        passed members to the newly created subclass of cls.

        This method helps to create enums having the same member values as
        values of other enum(s).

        :param name: name of the newly created type
        :param members: 1) a dict or 2) a list of (name, value) tuples
                        and/or EnumBase instances describing new members
        :return: newly created enum type.

        """
        NewEnum = type(name, (cls,), {})

        if isinstance(members, dict):
            members = members.items()
        for member in members:
            if isinstance(member, tuple):
                name, value = member
                setattr(NewEnum, name, value)
            elif isinstance(member, EnumBase):
                setattr(NewEnum, member.short_name, member.value)
            else:
                assert False, (
                    "members must be either a dict, "
                    + "a list of (name, value) tuples, "
                    + "or a list of EnumBase instances."
                )

        NewEnum.process()

        # needed for pickling to work (hopefully); taken from the namedtuple implementation in the
        # standard library
        try:
            NewEnum.__module__ = sys._getframe(1).f_globals.get("__name__", "__main__")
        except (AttributeError, ValueError):
            pass

        return NewEnum