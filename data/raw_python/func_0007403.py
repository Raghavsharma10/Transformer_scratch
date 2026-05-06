def _build_new_virtualenv(self):
        '''Build a new virtualenvironment if self._virtualenv is set to None'''
        if self._virtualenv is None:
            # virtualenv was "None" which means "do default"
            self._pkg_venv = os.path.join(self._temp_workspace, 'venv')
            self._venv_pip = 'bin/pip'
            if sys.platform == 'win32' or sys.platform == 'cygwin':
                self._venv_pip = 'Scripts\pip.exe'

            python_exe = self._python_executable()

            proc = Popen(["virtualenv", "-p", python_exe,
                          self._pkg_venv], stdout=PIPE, stderr=PIPE)
            stdout, stderr = proc.communicate()
            LOG.debug("Virtualenv stdout: %s" % stdout)
            LOG.debug("Virtualenv stderr: %s" % stderr)

            if proc.returncode is not 0:
                raise Exception('virtualenv returned unsuccessfully')

        else:
            raise Exception('cannot build a new virtualenv when asked to omit')