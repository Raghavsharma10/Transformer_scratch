def get_permissions(cls):
        """
        Generates permissions for all CrudView based class methods.

        Returns:
            List of Permission objects.
        """
        perms = []
        for kls_name, kls in cls.registry.items():
            for method_name in cls.__dict__.keys():
                if method_name.endswith('_view'):
                    perms.append("%s.%s" % (kls_name, method_name))
        return perms