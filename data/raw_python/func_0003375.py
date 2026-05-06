async def load_by_path(self, path):
        """
        Load a module by full path. If there are dependencies, they are also loaded.
        """
        try:
            p, module = findModule(path, True)
        except KeyError as exc:
            raise ModuleLoadException('Cannot load module ' + repr(path) + ': ' + str(exc) + 'is not defined in the package')
        except Exception as exc:
            raise ModuleLoadException('Cannot load module ' + repr(path) + ': ' + str(exc))
        if module is None:
            raise ModuleLoadException('Cannot find module: ' + repr(path))
        return await self.loadmodule(module)