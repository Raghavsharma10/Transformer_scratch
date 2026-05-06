def _validate(cls, group_name):
        """
        Validates the group name
        Input values must be strings (standard or unicode).  Throws ArgumentError if any input is invalid
        :param group_name: name of group
        """
        if group_name and not cls._group_name_regex.match(group_name):
            raise ArgumentError("'%s': Illegal group name" % (group_name,))
        if group_name and len(group_name) > 255:
            raise ArgumentError("'%s': Group name is too long" % (group_name,))