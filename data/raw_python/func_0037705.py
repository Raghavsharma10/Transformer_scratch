def get_obj_class(self, obj_type):
        """ Returns the object class based on parent and object types.

        In most cases the object class can be derived from object type alone but sometimes the
        same object type name is used for different object types so the parent (or even
        grandparent) type is required in order to determine the exact object type.
        For example, interface object type can be child of vport or router (ospf etc.). In the
        first case the required class is IxnInterface while in the later case it is IxnObject.

        :param obj_type: IXN object type.
        :return: object class if specific class else IxnObject.
        """

        if obj_type in IxnObject.str_2_class:
            if type(IxnObject.str_2_class[obj_type]) is dict:
                if self.obj_type() in IxnObject.str_2_class[obj_type]:
                    return IxnObject.str_2_class[obj_type][self.obj_type()]
                elif self.obj_parent().obj_type() in IxnObject.str_2_class[obj_type]:
                    return IxnObject.str_2_class[obj_type][self.obj_parent().obj_type()]
            else:
                return IxnObject.str_2_class[obj_type]
        return IxnObject