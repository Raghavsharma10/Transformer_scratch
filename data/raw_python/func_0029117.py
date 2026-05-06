def name_strip(self, name, is_policy=False, prefix=True):
        """
        Transforms name to AWS valid characters and adds prefix and type
        :param name: Name of the role/policy
        :param is_policy: True if policy should be added as suffix
        :param prefix: True if prefix should be added
        :return: Transformed and joined name
        """
        str = self.name_build(name, is_policy, prefix)
        str = str.title()
        str = str.replace('-', '')
        return str