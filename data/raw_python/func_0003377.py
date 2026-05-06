async def reload_modules(self, pathlist):
        """
        Reload modules with a full path in the pathlist
        """
        loadedModules = []
        failures = []
        for path in pathlist:
            p, module = findModule(path, False)
            if module is not None and hasattr(module, '_instance') and module._instance.state != ModuleLoadStateChanged.UNLOADED:
                loadedModules.append(module)
        # Unload all modules
        ums = [ModuleLoadStateChanged.createMatcher(m, ModuleLoadStateChanged.UNLOADED) for m in loadedModules]
        for m in loadedModules:
            # Only unload the module itself, not its dependencies, since we will restart the module soon enough
            self.subroutine(self.unloadmodule(m, True), False)
        await self.wait_for_all(*ums)
        # Group modules by package
        grouped = {}
        for path in pathlist:
            dotpos = path.rfind('.')
            if dotpos == -1:
                raise ModuleLoadException('Must specify module with full path, including package name')
            package = path[:dotpos]
            classname = path[dotpos + 1:]
            mlist = grouped.setdefault(package, [])
            p, module = findModule(path, False)
            mlist.append((classname, module))
        for package, mlist in grouped.items():
            # Reload each package only once
            try:
                p = sys.modules[package]
                # Remove cache to ensure a clean import from source file
                removeCache(p)
                p = reload(p)
            except KeyError:
                try:
                    p = __import__(package, fromlist=[m[0] for m in mlist])
                except Exception:
                    self._logger.warning('Failed to import a package: %r, resume others', package, exc_info = True)
                    failures.append('Failed to import: ' + package)
                    continue
            except Exception:
                self._logger.warning('Failed to import a package: %r, resume others', package, exc_info = True)
                failures.append('Failed to import: ' + package)
                continue                
            for cn, module in mlist:
                try:
                    module2 = getattr(p, cn)
                except AttributeError:
                    self._logger.warning('Cannot find module %r in package %r, resume others', package, cn)
                    failures.append('Failed to import: ' + package + '.' + cn)
                    continue
                if module is not None and module is not module2:
                    # Update the references
                    try:
                        lpos = loadedModules.index(module)
                        loaded = True
                    except Exception:
                        loaded = False
                    for d in module.depends:
                        # The new reference is automatically added on import, only remove the old reference
                        d.referencedBy.remove(module)
                        if loaded and hasattr(d, '_instance'):
                            try:
                                d._instance.dependedBy.remove(module)
                                d._instance.dependedBy.add(module2)
                            except ValueError:
                                pass
                    if hasattr(module, 'referencedBy'):
                        for d in module.referencedBy:
                            pos = d.depends.index(module)
                            d.depends[pos] = module2
                            if not hasattr(module2, 'referencedBy'):
                                module2.referencedBy = []
                            module2.referencedBy.append(d)
                    if loaded:
                        loadedModules[lpos] = module2
        # Start the uploaded modules
        for m in loadedModules:
            self.subroutine(self.loadmodule(m))
        if failures:
            raise ModuleLoadException('Following errors occurred during reloading, check log for more details:\n' + '\n'.join(failures))