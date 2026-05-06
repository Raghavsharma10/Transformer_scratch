def newEmailReport(self, name, **kwargs):
        """Creates a new email report

        Returns status message for operation

        Optional parameters:

            * checkid -- Check identifier. If omitted, this will be an
                overview report
                    Type: Integer

            * frequency -- Report frequency
                    Type: String ['monthly', 'weekly', 'daily']

            * contactids -- Comma separated list of receiving contact
                identifiers
                    Type: String

            * additionalemails -- Comma separated list of additional receiving
                emails
                    Type: String
        """

        # Warn user about unhandled parameters
        for key in kwargs:
            if key not in ['checkid', 'frequency', 'contactids',
                           'additionalemails']:
                sys.stderr.write("'%s'" % key + ' is not a valid argument ' +
                                 'of newEmailReport()\n')

        parameters = {'name': name}
        for key, value in kwargs.iteritems():
            parameters[key] = value

        return self.request('POST', 'reports.email',
                            parameters).json()['message']