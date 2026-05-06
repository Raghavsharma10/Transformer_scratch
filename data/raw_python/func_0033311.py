def _input_as_list(self, data):
        '''Takes the positional arguments as input in a list.

        The list input here should be [query_file_path, database_file_path,
        output_file_path]'''
        query, database, output = data
        if (not isabs(database)) \
                or (not isabs(query)) \
                or (not isabs(output)):
            raise ApplicationError("Only absolute paths allowed.\n%s" %
                                   ', '.join(data))

        self._database = FilePath(database)
        self._query = FilePath(query)
        self._output = ResultPath(output, IsWritten=True)

        # check parameters that can only take a particular set of values
        # check combination of databse and query type
        if self.Parameters['-t'].isOn() and self.Parameters['-q'].isOn() and \
                (self.Parameters['-t'].Value, self.Parameters['-q'].Value) not in \
                self._valid_combinations:
            error_message = "Invalid combination of database and query " + \
                            "types ('%s', '%s').\n" % \
                            (self.Paramters['-t'].Value,
                             self.Parameters['-q'].Value)

            error_message += "Must be one of: %s\n" % \
                             repr(self._valid_combinations)

            raise ApplicationError(error_message)

        # check database type
        if self.Parameters['-t'].isOn() and \
                self.Parameters['-t'].Value not in self._database_types:
            error_message = "Invalid database type %s\n" % \
                            self.Parameters['-t'].Value

            error_message += "Allowed values: %s\n" % \
                             ', '.join(self._database_types)

            raise ApplicationError(error_message)

        # check query type
        if self.Parameters['-q'].isOn() and \
                self.Parameters['-q'].Value not in self._query_types:
            error_message = "Invalid query type %s\n" % \
                            self.Parameters['-q'].Value

            error_message += "Allowed values: %s\n" % \
                ', '.join(self._query_types)

            raise ApplicationError(error_message)

        # check mask type
        if self.Parameters['-mask'].isOn() and \
                self.Parameters['-mask'].Value not in self._mask_types:
            error_message = "Invalid mask type %s\n" % \
                            self.Parameters['-mask']

            error_message += "Allowed Values: %s\n" % \
                ', '.join(self._mask_types)

            raise ApplicationError(error_message)

        # check qmask type
        if self.Parameters['-qMask'].isOn() and \
                self.Parameters['-qMask'].Value not in self._mask_types:
            error_message = "Invalid qMask type %s\n" % \
                            self.Parameters['-qMask'].Value

            error_message += "Allowed values: %s\n" % \
                             ', '.join(self._mask_types)

            raise ApplicationError(error_message)

        # check repeat type
        if self.Parameters['-repeats'].isOn() and \
                self.Parameters['-repeats'].Value not in self._mask_types:
            error_message = "Invalid repeat type %s\n" % \
                            self.Parameters['-repeat'].Value

            error_message += "Allowed values: %s\n" % \
                             ', '.join(self._mask_types)

            raise ApplicationError(error_message)

        # check output format
        if self.Parameters['-out'].isOn() and \
                self.Parameters['-out'].Value not in self._out_types:
            error_message = "Invalid output type %s\n" % \
                            self.Parameters['-out']

            error_message += "Allowed values: %s\n" % \
                             ', '.join(self._out_types)

            raise ApplicationError(error_message)

        return ''