def _buildElementTree(self,):
        """Turn object into an ElementTree
        """
        t_elt = ctree.Element(self.name)

        for k,v in [ (key,value) for key,value in self.__dict__.items() if key != 'name']: # Excluding name from list of items
            if v and v != 'false' :
                t_elt.set(k if k != 'like' else 'as', str(v).lower())

        self._etree = t_elt
        return t_elt