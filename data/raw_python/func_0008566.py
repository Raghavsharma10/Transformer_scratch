def binary_size(self):
        '''Return the number of bytes needed to store this parameter.'''
        return (
            1 + # group_id
            2 + # next offset marker
            1 + len(self.name.encode('utf-8')) + # size of name and name bytes
            1 + # data size
            1 + len(self.dimensions) + # size of dimensions and dimension bytes
            self.total_bytes + # data
            1 + len(self.desc.encode('utf-8')) # size of desc and desc bytes
            )