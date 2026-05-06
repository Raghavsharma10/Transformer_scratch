def _install_requirements(self):
        '''
        Create a new virtualenvironment and install requirements
        if there are any.
        '''
        if not hasattr(self, '_pkg_venv'):
            err = 'Must call build_new_virtualenv before install_requirements'
            raise Exception(err)

        cmd = None
        if self._requirements:
            LOG.debug("Installing requirements found %s in config"
                      % self._requirements)
            cmd = [os.path.join(self._pkg_venv, self._venv_pip),
                   'install'] + self._requirements

        elif _isfile(self._requirements_file):
            # Pip install
            LOG.debug("Installing requirements from requirements.txt file")
            cmd = [os.path.join(self._pkg_venv, self._venv_pip),
                   "install", "-r",
                   self._requirements_file]

        if cmd is not None:
            prc = Popen(cmd, stdout=PIPE, stderr=PIPE)
            stdout, stderr = prc.communicate()
            LOG.debug("Pip stdout: %s" % stdout)
            LOG.debug("Pip stderr: %s" % stderr)

            if prc.returncode is not 0:
                raise Exception('pip returned unsuccessfully')