def get_group_instance(self, parent):
        """Create an instance object"""
        o = copy.copy(self)
        o.init_instance(parent)
        return o