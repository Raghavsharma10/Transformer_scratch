def is_owned_by_current_user(self):
        """ Check if the current user owns the object """

        from bambou.nurest_root_object import NURESTRootObject
        root_object = NURESTRootObject.get_default_root_object()
        return self._owner == root_object.id