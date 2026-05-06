def get_role_keys(cls, unit_key):
        """
        :param unit_key: Parent unit key
        :return: role keys of subunits
        """
        stack = Role.objects.filter(unit_id=unit_key).values_list('key', flatten=True)
        for unit_key in cls.objects.filter(parent_id=unit_key).values_list('key', flatten=True):
            stack.extend(cls.get_role_keys(unit_key))
        return stack