def _input_as_dict(self, data):
        """Takes dictionary that sets input and output files.

        Valid keys for the dictionary are specified in the subclasses. File
        paths must be absolute.
        """
        # clear self._input; ready to receive new input and output files
        self._input = {}
        # Check that the arguments to the
        # subcommand-specific parameters are valid
        self.check_arguments()

        # Ensure that we have all required input (file I/O)
        for k in self._input_order:
            # N.B.: optional positional arguments begin with underscore (_)!
            # (e.g., see _mate_in for bwa bwasw)
            if k[0] != '_' and k not in data:
                raise MissingRequiredArgumentApplicationError("Missing "
                                                              "required "
                                                              "input %s" % k)

        # Set values for input and output files
        for k in data:
            # check for unexpected keys in the dict
            if k not in self._input_order:
                error_message = "Invalid input arguments (%s)\n" % k
                error_message += "Valid keys are: %s" % repr(self._input_order)
                raise InvalidArgumentApplicationError(error_message + '\n')

            # check for absolute paths
            if not isabs(data[k][0]):
                raise InvalidArgumentApplicationError("Only absolute paths "
                                                      "allowed.\n%s" %
                                                      repr(data))
            self._input[k] = data[k]

        # if there is a -f option to specify an output file, force the user to
        # use it (otherwise things to to stdout)
        if '-f' in self.Parameters and not self.Parameters['-f'].isOn():
            raise InvalidArgumentApplicationError("Please specify an output "
                                                  "file with -f")

        return ''