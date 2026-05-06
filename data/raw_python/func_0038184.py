def add(self, parent, obj_type, **attributes):
        """ IXN API add command

        @param parent: object parent - object will be created under this parent.
        @param object_type: object type.
        @param attributes: additional attributes.
        @return: IXN object reference.
        """
        return self.ixnCommand('add', parent.obj_ref(), obj_type, get_args_pairs(attributes))