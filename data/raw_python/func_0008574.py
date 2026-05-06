def write(self, group_id, handle):
        '''Write this parameter group, with parameters, to a file handle.

        Parameters
        ----------
        group_id : int
            The numerical ID of the group.
        handle : file handle
            An open, writable, binary file handle.
        '''
        name = self.name.encode('utf-8')
        desc = self.desc.encode('utf-8')
        handle.write(struct.pack('bb', len(name), -group_id))
        handle.write(name)
        handle.write(struct.pack('<h', 3 + len(desc)))
        handle.write(struct.pack('B', len(desc)))
        handle.write(desc)
        for param in self.params.values():
            param.write(group_id, handle)