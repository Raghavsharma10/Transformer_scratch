def _error_on_missing_application(self, params):
        """ Raise an ApplicationNotFoundError if the app is not accessible

            This method checks in the system path (usually $PATH) or for
            the existence of self._command. If self._command is not found
            in either place, an ApplicationNotFoundError is raised to
            inform the user that the application they are trying to access is
            not available.

            This method should be overwritten when self._command does not
            represent the relevant executable (e.g., self._command = 'prog -a')
            or in more complex cases where the file to be executed may be
            passed as a parameter (e.g., with java jar files, where the
            jar file is passed to java via '-jar'). It can also be overwritten
            to by-pass testing for application presence by never raising an
            error.
        """
        command = self._command
        # strip off " characters, in case we got a FilePath object
        found_in_path = which(command.strip('"')) is not None
        if not (exists(command) or found_in_path):
            raise ApplicationNotFoundError("Cannot find %s. Is it installed? "
                                           "Is it in your path?" % command)