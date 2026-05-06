def newSharedReport(self, checkid, **kwargs):
        """Create a shared report (banner).

        Returns status message for operation

        Optional parameters:

            * auto -- Automatic period (If false, requires: fromyear,
                frommonth, fromday, toyear, tomonth, today)
                    Type: Boolean

            * type -- Banner type
                    Type: String ['uptime', 'response']

            * fromyear -- Period start: year
                    Type: Integer

            * frommonth -- Period start: month
                    Type: Integer

            * fromday -- Period start: day
                    Type: Integer

            * toyear -- Period end: year
                    Type: Integer

            * tomonth -- Period end: month
                    Type: Integer

            * today -- Period end: day
                    Type: Integer
        """

        # Warn user about unhandled parameters
        for key in kwargs:
            if key not in ['auto', 'type', 'fromyear', 'frommonth', 'fromday',
                           'toyear', 'tomonth', 'today', 'sharedtype']:
                sys.stderr.write("'%s'" % key + ' is not a valid argument ' +
                                 'of newSharedReport()\n')

        parameters = {'checkid': checkid, 'sharedtype': 'banner'}
        for key, value in kwargs.iteritems():
            parameters[key] = value

        return self.request('POST', 'reports.shared',
                            parameters).json()['message']