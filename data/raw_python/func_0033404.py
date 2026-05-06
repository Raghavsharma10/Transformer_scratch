def add_namespace(self, *args, **kwargs):
        """Accepts two calling patterns:
        add_namespace(namespace): queue a preexisting namespace onto
            this VW instance.
        add_namespace(name, scale, features, ...): Pass all args and kwargs
            to the Namespace constructor to make a new Namespace instance,
            and queue it to this VW instance.

        Returns self (so that this command can be chained).
        """
        if args and isinstance(args[0], Namespace):
            namespace = args[0]
        elif isinstance(kwargs.get('namespace'), Namespace):
            namespace = kwargs.get('namespace')
        else:
            namespace = Namespace(*args, **kwargs)
        self.namespaces.append(namespace)
        return self