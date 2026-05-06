def get_children(self, *types):
        """ Read (getList) children from IXN.

        Use this method to align with current IXN configuration.

        :param types: list of requested children.
        :return: list of all children objects of the requested types.
        """

        children_objs = OrderedDict()
        if not types:
            types = self.get_all_child_types(self.obj_ref())
        for child_type in types:
            children_list = self.api.getList(self.obj_ref(), child_type)
            children_objs.update(self._build_children_objs(child_type, children_list))
        return list(children_objs.values())