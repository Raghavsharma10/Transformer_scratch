def virtualenv(self, virtualenv):
        '''
        Sets the virtual environment for the lambda package

        If this is not set then package_dependencies will create a new one.

        Takes a path to a virtualenv or a boolean if the virtualenv creation
        should be skipped.
        '''
        # If a boolean is passed then set the internal _skip_virtualenv flag
        if isinstance(virtualenv, bool):
            self._skip_virtualenv = virtualenv
        else:
            self._virtualenv = virtualenv
            if not os.path.isdir(self._virtualenv):
                raise Exception("virtualenv %s not found" % self._virtualenv)
            LOG.info("Using existing virtualenv at %s" % self._virtualenv)
            # use supplied virtualenv path
            self._pkg_venv = self._virtualenv
            self._skip_virtualenv = True