def parameter_blocks(self):
        '''Compute the size (in 512B blocks) of the parameter section.'''
        bytes = 4. + sum(g.binary_size() for g in self.groups.values())
        return int(np.ceil(bytes / 512))