def requirements(self, requires):
        '''
        Sets the requirements for the package.

        It will take either a valid path to a requirements file or
        a list of requirements.
        '''
        if requires:
            if isinstance(requires, basestring) and \
               os.path.isfile(os.path.abspath(requires)):
                self._requirements_file = os.path.abspath(requires)
            else:
                if isinstance(self._requirements, basestring):
                    requires = requires.split()
                self._requirements_file = None
                self._requirements = requires
        else:
            # If the default requirements file is found use that
            if os.path.isfile(self._requirements_file):
                return
            self._requirements, self._requirements_file = None, None