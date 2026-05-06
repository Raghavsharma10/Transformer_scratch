def concept(self, cid, **kwargs):
        ''' Get concept by concept ID '''
        if cid not in self.__concept_map:
            if 'default' in kwargs:
                return kwargs['default']
            else:
                raise KeyError("Invalid cid")
        else:
            return self.__concept_map[cid]