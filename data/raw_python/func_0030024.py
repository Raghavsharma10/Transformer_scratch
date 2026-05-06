def mangle_name(name):
        """Mangles a column name to a standard form, remoing illegal
        characters.

        :param name:
        :return:

        """
        import re
        try:
            return re.sub('_+', '_', re.sub('[^\w_]', '_', name).lower()).rstrip('_')
        except TypeError:
            raise TypeError(
                'Trying to mangle name with invalid type of: ' + str(type(name)))