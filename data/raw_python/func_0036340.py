def is_nameserver(self, path):
        '''Is the node pointed to by @ref path a name server (specialisation
        of directory nodes)?

        '''
        node = self.get_node(path)
        if not node:
            return False
        return node.is_nameserver