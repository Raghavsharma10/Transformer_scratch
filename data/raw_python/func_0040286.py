def get_methodnames(self, node):
        '''Given a node, generate all names for matching visitor methods.
        '''
        nodekey = self.get_nodekey(node)
        prefix = self._method_prefix
        if isinstance(nodekey, self.GeneratorType):
            for nodekey in nodekey:
                yield self._method_prefix + nodekey
        else:
            yield self._method_prefix + nodekey