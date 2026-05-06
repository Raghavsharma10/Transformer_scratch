def addToService(self, service, namespace=None, seperator='.'):
        """
        Add this Handler's exported methods to an RPC Service instance.
        """
        if namespace is None:
            namespace = []
        if isinstance(namespace, basestring):
            namespace = [namespace]

        for n, m in inspect.getmembers(self, inspect.ismethod):
            if hasattr(m, 'export_rpc'):
                try:
                    name = seperator.join(namespace + m.export_rpc)
                except TypeError:
                    name = seperator.join(namespace + [m.export_rpc])
                service.add(m, name)