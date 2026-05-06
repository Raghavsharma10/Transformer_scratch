def _set_options_from_file(self, file_handle):
        """Parses a unit file and updates self._data['options']

        Args:
            file_handle (file): a file-like object (supporting read()) containing a unit

        Returns:
            True: The file was successfuly parsed and options were updated

        Raises:
            IOError: from_file was specified and it does not exist
            ValueError: The unit contents specified in from_string or from_file is not valid
        """

        # TODO: Find a library to handle this unit file parsing
        # Can't use configparser, it doesn't handle multiple entries for the same key in the same section
        # This is terribly naive

        # build our output here
        options = []

        # keep track of line numbers to report when parsing problems happen
        line_number = 0

        # the section we are currently in
        section = None
        for line in file_handle.read().splitlines():
            line_number += 1

            # clear any extra white space
            orig_line = line
            line = line.strip()

            # ignore comments, and blank lines
            if not line or line.startswith('#'):
                continue

            # is this a section header?  If so, update our variable and continue
            # Section headers look like: [Section]
            if line.startswith('[') and line.endswith(']'):
                section = line.strip('[]')
                continue

            # We encountered a non blank line outside of a section, this is a problem
            if not section:
                raise ValueError(
                    'Unable to parse unit file; '
                    'Unexpected line outside of a section: {0} (line: {1}'.format(
                        line,
                        line_number
                    ))

            # Attempt to parse a line inside a section
            # Lines should look like: name=value \
            # continuation
            continuation = False
            try:
                    # if the previous value ends with \ then we are a continuation
                    # so remove the \, and set the flag so we'll append to this below
                    if options[-1]['value'].endswith('\\'):
                        options[-1]['value'] = options[-1]['value'][:-1]
                        continuation = True
            except IndexError:
                    pass

            try:
                # if we are a continuation, then just append our value to the previous line
                if continuation:
                    options[-1]['value'] += orig_line
                    continue

                # else we are a normal line, so spit and get our name / value
                name, value = line.split('=', 1)
                options.append({
                    'section': section,
                    'name': name,
                    'value': value
                })
            except ValueError:
                raise ValueError(
                    'Unable to parse unit file; '
                    'Malformed line in section {0}: {1} (line: {2})'.format(
                        section,
                        line,
                        line_number
                    ))

        # update our internal structure
        self._data['options'] = options

        return True