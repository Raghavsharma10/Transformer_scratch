async def loadmodule(self, module):
        '''
        Load a module class
        '''
        self._logger.debug('Try to load module %r', module)
        if hasattr(module, '_instance'):
            self._logger.debug('Module is already initialized, module state is: %r', module._instance.state)
            if module._instance.state == ModuleLoadStateChanged.UNLOADING or module._instance.state == ModuleLoadStateChanged.UNLOADED:
                # Wait for unload
                um = ModuleLoadStateChanged.createMatcher(module._instance.target, ModuleLoadStateChanged.UNLOADED)
                await um
            elif module._instance.state == ModuleLoadStateChanged.SUCCEEDED:
                pass
            elif module._instance.state == ModuleLoadStateChanged.FAILED:
                raise ModuleLoadException('Cannot load module %r before unloading the failed instance' % (module,))
            elif module._instance.state == ModuleLoadStateChanged.NOTLOADED or module._instance.state == ModuleLoadStateChanged.LOADED or module._instance.state == ModuleLoadStateChanged.LOADING:
                # Wait for succeeded or failed
                sm = ModuleLoadStateChanged.createMatcher(module._instance.target, ModuleLoadStateChanged.SUCCEEDED)
                fm = ModuleLoadStateChanged.createMatcher(module._instance.target, ModuleLoadStateChanged.FAILED)
                _, m = await M_(sm, fm)
                if m is sm:
                    pass
                else:
                    raise ModuleLoadException('Module load failed for %r' % (module,))
        if not hasattr(module, '_instance'):
            self._logger.info('Loading module %r', module)
            inst = module(self.server)
            # Avoid duplicated loading
            module._instance = inst
            # When reloading, some of the dependencies are broken, repair them
            if hasattr(module, 'referencedBy'):
                inst.dependedBy = set([m for m in module.referencedBy if hasattr(m, '_instance') and m._instance.state != ModuleLoadStateChanged.UNLOADED])
            # Load Module
            for d in module.depends:
                if hasattr(d, '_instance') and d._instance.state == ModuleLoadStateChanged.FAILED:
                    raise ModuleLoadException('Cannot load module %r, it depends on %r, which is in failed state.' % (module, d))
            try:
                for d in module.depends:
                    if hasattr(d, '_instance') and d._instance.state == ModuleLoadStateChanged.SUCCEEDED:
                        d._instance.dependedBy.add(module)
                preloads = [d for d in module.depends if not hasattr(d, '_instance') or \
                            d._instance.state != ModuleLoadStateChanged.SUCCEEDED]
                for d in preloads:
                    self.subroutine(self.loadmodule(d), False)
                sms = [ModuleLoadStateChanged.createMatcher(d, ModuleLoadStateChanged.SUCCEEDED) for d in preloads]
                fms = [ModuleLoadStateChanged.createMatcher(d, ModuleLoadStateChanged.FAILED) for d in preloads]
                def _callback(event, matcher):
                    raise ModuleLoadException('Cannot load module %r, it depends on %r, which is in failed state.' % (module, event.module))
                await self.with_callback(self.wait_for_all(*sms), _callback, *fms)
            except:
                for d in module.depends:
                    if hasattr(d, '_instance') and module in d._instance.dependedBy:
                        self._removeDepend(module, d)
                self._logger.warning('Loading module %r stopped', module, exc_info=True)
                # Not loaded, send a message to notify that parents can not 
                async def _msg():
                    await self.wait_for_send(ModuleLoadStateChanged(module, ModuleLoadStateChanged.UNLOADED, inst))
                self.subroutine(_msg(), False)
                del module._instance
                raise
            for d in preloads:
                d._instance.dependedBy.add(module)
            self.activeModules[inst.getServiceName()] = module
            await module._instance.changestate(ModuleLoadStateChanged.LOADING, self)
            await module._instance.load(self)
            if module._instance.state == ModuleLoadStateChanged.LOADED or module._instance.state == ModuleLoadStateChanged.LOADING:
                sm = ModuleLoadStateChanged.createMatcher(module._instance.target, ModuleLoadStateChanged.SUCCEEDED)
                fm = ModuleLoadStateChanged.createMatcher(module._instance.target, ModuleLoadStateChanged.FAILED)
                _, m = await M_(sm, fm)
                if m is sm:
                    pass
                else:
                    raise ModuleLoadException('Module load failed for %r' % (module,))
            self._logger.info('Loading module %r completed, module state is %r', module, module._instance.state)
            if module._instance.state == ModuleLoadStateChanged.FAILED:
                raise ModuleLoadException('Module load failed for %r' % (module,))