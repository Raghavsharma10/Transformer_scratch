def install_dependencies(self):
        ''' Creates a virtualenv and installs requirements '''
        # If virtualenv is set to skip then do nothing
        if self._skip_virtualenv:
            LOG.info('Skip Virtualenv set ... nothing to do')
            return

        has_reqs = _isfile(self._requirements_file) or self._requirements
        if self._virtualenv is None and has_reqs:
            LOG.info('Building new virtualenv and installing requirements')
            self._build_new_virtualenv()
            self._install_requirements()
        elif self._virtualenv is None and not has_reqs:
            LOG.info('No requirements found, so no virtualenv will be made')
            self._pkg_venv = False
        else:
            raise Exception('Cannot determine what to do about virtualenv')