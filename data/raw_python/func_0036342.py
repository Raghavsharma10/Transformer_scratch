def is_zombie(self, path):
        '''Is the node pointed to by @ref path a zombie object?'''
        node = self.get_node(path)
        if not node:
            return False
        return node.is_zombie