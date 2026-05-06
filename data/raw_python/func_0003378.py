def get_module_by_name(self, targetname):
        """
        Return the module instance for a target name.
        """
        if targetname == 'public':
            target = None
        elif not targetname not in self.activeModules:
            raise KeyError('Module %r not exists or is not loaded' % (targetname,))
        else:
            target = self.activeModules[targetname]
        return target