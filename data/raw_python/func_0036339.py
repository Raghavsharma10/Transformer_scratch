def is_manager(self, path):
        '''Is the node pointed to by @ref path a manager?'''
        node = self.get_node(path)
        if not node:
            return False
        return node.is_manager