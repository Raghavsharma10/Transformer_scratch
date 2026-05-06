def _validate_inputs(self, inputdict):
        """ Validate input links.
        """
        # Check inputdict
        try:
            parameters = inputdict.pop(self.get_linkname('parameters'))
        except KeyError:
            raise InputValidationError("No parameters specified for this "
                                       "calculation")
        if not isinstance(parameters, RipsDistanceMatrixParameters):
            raise InputValidationError("parameters not of type "
                                       "RipsDistanceMatrixParameters")
        # Check code
        try:
            code = inputdict.pop(self.get_linkname('code'))
        except KeyError:
            raise InputValidationError("No code specified for this "
                                       "calculation")

        # Check input files
        try:
            distance_matrix = inputdict.pop(
                self.get_linkname('distance_matrix'))
            if not isinstance(distance_matrix, SinglefileData):
                raise InputValidationError(
                    "distance_matrix not of type SinglefileData")
            symlink = None

        except KeyError:
            distance_matrix = None

            try:
                remote_folder = inputdict.pop(
                    self.get_linkname('remote_folder'))
                if not isinstance(remote_folder, RemoteData):
                    raise InputValidationError(
                        "remote_folder is not of type RemoteData")

                comp_uuid = remote_folder.get_computer().uuid
                remote_path = remote_folder.get_remote_path()
                symlink = (comp_uuid, remote_path, self._REMOTE_FOLDER_LINK)

            except KeyError:
                raise InputValidationError(
                    "Need to provide either distance_matrix or remote_folder")

        # Check that nothing is left unparsed
        if inputdict:
            raise ValidationError("Unrecognized inputs: {}".format(inputdict))

        return parameters, code, distance_matrix, symlink