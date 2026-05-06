def is_directory(self, path):
        '''Is the node pointed to by @ref path a directory (name servers and
        naming contexts)?

        '''
        node = self.get_node(path)
        if not node:
            return False
        return node.is_directory