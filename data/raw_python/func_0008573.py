def binary_size(self):
        '''Return the number of bytes to store this group and its parameters.'''
        return (
            1 + # group_id
            1 + len(self.name.encode('utf-8')) + # size of name and name bytes
            2 + # next offset marker
            1 + len(self.desc.encode('utf-8')) + # size of desc and desc bytes
            sum(p.binary_size() for p in self.params.values()))