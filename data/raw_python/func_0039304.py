def _buildElementTree(self,):
        """Turn object into an Element Tree
        """

        t_paging = ctree.Element('paging')
        t_paging.set('model', self.model)
        for key in self.__dict__.keys():
            if key != 'model':
                t_tag = ctree.SubElement(t_paging, key)
                for item in self.__dict__[key].items():
                    t_tag.set(*item)

        self.etree = t_paging
        return  t_paging