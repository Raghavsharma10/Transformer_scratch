def _gatherPluginMethods(self, methodName):
        """
        Walk through each L{IOrganizerPlugin} powerup, yielding the bound
        method if the powerup implements C{methodName}.  Upon encountering a
        plugin which fails to implement it, issue a
        L{PendingDeprecationWarning}.

        @param methodName: The name of a L{IOrganizerPlugin} method.
        @type methodName: C{str}

        @return: Iterable of methods.
        """
        for plugin in self.getOrganizerPlugins():
            implementation = getattr(plugin, methodName, None)
            if implementation is not None:
                yield implementation
            else:
                warn(
                    ('IOrganizerPlugin now has the %r method, %s'
                        ' did not implement it') % (
                            methodName, plugin.__class__),
                    category=PendingDeprecationWarning)