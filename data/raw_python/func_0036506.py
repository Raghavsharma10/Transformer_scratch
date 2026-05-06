def unbind(self, name):
        '''Unbind an object from the context represented by this directory.

        Warning: this is a dangerous operation. You may unlink an entire
        section of the tree and be unable to recover it. Be careful what you
        unbind.

        The name should be in the format used in paths. For example,
        'manager.mgr' or 'ConsoleIn0.rtc'.

        '''
        with self._mutex:
            id, sep, kind = name.rpartition('.')
            if not id:
                id = kind
                kind = ''
            name = CosNaming.NameComponent(id=str(id), kind=str(kind))
            try:
                self.context.unbind([name])
            except CosNaming.NamingContext.NotFound:
                raise exceptions.BadPathError(name)