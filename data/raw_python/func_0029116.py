def name_build(self, name, is_policy=False, prefix=True):
        """
        Build name from prefix and name + type
        :param name: Name of the role/policy
        :param is_policy: True if policy should be added as suffix
        :param prefix: True if prefix should be added
        :return: Joined name
        """
        str = name

        # Add prefix
        if prefix:
            str = self.__role_name_prefix + str

        # Add policy suffix
        if is_policy:
            str = str + "-policy"

        return str