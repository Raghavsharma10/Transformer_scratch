def is_unknown(self, path):
        '''Is the node pointed to by @ref path an unknown object?'''
        node = self.get_node(path)
        if not node:
            return True
        return node.is_unknown