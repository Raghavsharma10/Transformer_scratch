def add(self, parent, obj_type, **attributes):
        """ IXN API add command

        @param parent: object parent - object will be created under this parent.
        @param object_type: object type.
        @param attributes: additional attributes.
        @return: IXN object reference.
        """

        return self.ixn.add(parent.obj_ref(), obj_type, *self._get_args_list(**attributes))