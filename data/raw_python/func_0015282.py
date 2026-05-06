def _install_dependencies(self, ui, debug):
        """Install missing dependencies"""
        for dep_t, dep_l in self.dependencies:
            if not dep_l:
                continue
            pkg_mgr = self.get_package_manager(dep_t)
            pkg_mgr.works()
            to_resolve = []
            for dep in dep_l:
                if not pkg_mgr.is_pkg_installed(dep):
                    to_resolve.append(dep)
            if not to_resolve:
                # nothing to install, let's move on
                continue
            to_install = pkg_mgr.resolve(*to_resolve)
            confirm = self._ask_to_confirm(ui, pkg_mgr, *to_install)
            if not confirm:
                msg = 'List of packages denied by user, exiting.'
                raise exceptions.DependencyException(msg)

            type(self).install_lock = True
            # TODO: we should do this more systematically (send signal to cl/gui?)
            logger.info('Installing dependencies, sit back and relax ...',
                        extra={'event_type': 'dep_installation_start'})
            if ui == 'cli' and not debug:  # TODO: maybe let every manager to decide when to start
                event = threading.Event()
                t = EndlessProgressThread(event)
                t.start()
            installed = pkg_mgr.install(*to_install)
            if ui == 'cli' and not debug:
                event.set()
                t.join()
                if installed:
                    logger.info(' Done.')
                else:
                    logger.error(' Failed.')
            type(self).install_lock = False

            log_extra = {'event_type': 'dep_installation_end'}
            if not installed:
                msg = 'Failed to install dependencies, exiting.'
                logger.error(msg, extra=log_extra)
                raise exceptions.DependencyException(msg)
            else:
                logger.info('Successfully installed dependencies!', extra=log_extra)