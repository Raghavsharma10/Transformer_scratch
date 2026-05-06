def validate_permission(self, key, permission):
        """ validates if group can get assigned with permission"""
        if permission.perm_name not in self.__possible_permissions__:
            raise AssertionError(
                "perm_name is not one of {}".format(self.__possible_permissions__)
            )
        return permission