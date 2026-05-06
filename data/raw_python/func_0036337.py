def is_component(self, path):
        '''Is the node pointed to by @ref path a component?'''
        node = self.get_node(path)
        if not node:
            return False
        return node.is_component