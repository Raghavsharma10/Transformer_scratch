def _buildElementTree(self,):
        """Turns object into a Element Tree
        """
        t_binder = ctree.Element(self.name)

        for k,v in self.__dict__.items():
            if k not in ('name', 'urls', 'inputs', 'paging') and v :
                t_binder.set(k,v)

        self.etree = t_binder
        return t_binder