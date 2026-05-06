async def unloadmodule(self, module, ignoreDependencies = False):
        '''
        Unload a module class
        '''
        self._logger.debug('Try to unload module %r', module)
        if hasattr(module, '_instance'):
            self._logger.debug('Module %r is loaded, module state is %r', module, module._instance.state)
            inst = module._instance
            if inst.state == ModuleLoadStateChanged.LOADING or inst.state == ModuleLoadStateChanged.LOADED:
                # Wait for loading
                # Wait for succeeded or failed
                sm = ModuleLoadStateChanged.createMatcher(module._instance.target, ModuleLoadStateChanged.SUCCEEDED)
                fm = ModuleLoadStateChanged.createMatcher(module._instance.target, ModuleLoadStateChanged.FAILED)
                await M_(sm, fm)
            elif inst.state == ModuleLoadStateChanged.UNLOADING or inst.state == ModuleLoadStateChanged.UNLOADED:
                um = ModuleLoadStateChanged.createMatcher(module, ModuleLoadStateChanged.UNLOADED)
                await um
        if hasattr(module, '_instance') and (module._instance.state == ModuleLoadStateChanged.SUCCEEDED or
                                             module._instance.state == ModuleLoadStateChanged.FAILED):
            self._logger.info('Unloading module %r', module)
            inst = module._instance
            # Change state to unloading to prevent more dependencies
            await inst.changestate(ModuleLoadStateChanged.UNLOADING, self)
            if not ignoreDependencies:
                deps = [d for d in inst.dependedBy if hasattr(d, '_instance') and d._instance.state != ModuleLoadStateChanged.UNLOADED]
                ums = [ModuleLoadStateChanged.createMatcher(d, ModuleLoadStateChanged.UNLOADED) for d in deps]
                for d in deps:
                    self.subroutine(self.unloadmodule(d), False)
                await self.wait_for_all(*ums)
            await inst.unload(self)
            del self.activeModules[inst.getServiceName()]
            self._logger.info('Module %r is unloaded', module)
            if not ignoreDependencies:
                for d in module.depends:
                    if hasattr(d, '_instance') and module in d._instance.dependedBy:
                        self._removeDepend(module, d)