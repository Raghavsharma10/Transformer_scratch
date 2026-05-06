def search(self, spec, operator):
        '''Query PYPI via XMLRPC interface using search spec'''
        return self.xmlrpc.search(spec, operator.lower())