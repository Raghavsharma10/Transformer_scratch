def _find_local_handlers(cls, handlers,  namespace, configs):
        """Add name info to every "local" (present in the body of this class)
        handler and add it to the mapping.
        """
        for aname, avalue in namespace.items():
            sig_name, config = cls._is_handler(aname, avalue)
            if sig_name:
                configs[aname] = config
                handlers[aname] = sig_name