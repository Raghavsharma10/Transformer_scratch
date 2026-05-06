async def unload_by_path(self, path):
        """
        Unload a module by full path. Dependencies are automatically unloaded if they are marked to be
        services.
        """
        p, module = findModule(path, False)
        if module is None:
            raise ModuleLoadException('Cannot find module: ' + repr(path))
        return await self.unloadmodule(module)