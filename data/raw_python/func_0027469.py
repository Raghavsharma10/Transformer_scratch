def actions(self, **parameters):
        """Returns a list of actions (alerts) that have been generated for
            your account.

        Optional Parameters:

            * from -- Only include actions generated later than this timestamp.
                Format is UNIX time.
                    Type: Integer
                    Default: None

            * to -- Only include actions generated prior to this timestamp.
                Format is UNIX time.
                    Type: Integer
                    Default: None

            * limit -- Limits the number of returned results to the specified
                quantity.
                    Type: Integer (max 300)
                    Default: 100

            * offset -- Offset for listing.
                    Type: Integer
                    Default: 0

            * checkids -- Comma-separated list of check identifiers. Limit
                results to actions generated from these checks.
                    Type: String
                    Default: All

            * contactids -- Comma-separated list of contact identifiers.
                Limit results to actions sent to these contacts.
                    Type: String
                    Default: All

            * status -- Comma-separated list of statuses. Limit results to
                actions with these statuses.
                    Type: String ['sent', 'delivered', 'error',
                        'not_delivered', 'no_credits']
                    Default: All

            * via -- Comma-separated list of via mediums. Limit results to
                actions with these mediums.
                    Type: String ['email', 'sms', 'twitter', 'iphone',
                        'android']
                    Default: All

        Returned structure:
        {
            'alerts' : [
                {
                    'contactname' : <String> Name of alerted contact
                    'contactid'   : <String> Identifier of alerted contact
                    'checkid'     : <String> Identifier of check
                    'time'        : <Integer> Time of alert generation. Format
                                              UNIX time
                    'via'         : <String> Alert medium ['email', 'sms',
                                                           'twitter', 'iphone',
                                                           'android']
                    'status'      : <String> Alert status ['sent', 'delivered',
                                                           'error',
                                                           'notdelivered',
                                                           'nocredits']
                    'messageshort': <String> Short description of message

                    'messagefull' : <String> Full message body
                    'sentto'      : <String> Target address, phone number, etc
                    'charged'     : <Boolean> True if your account was charged
                                              for this message
                },
                ...
            ]
        }
        """

        # Warn user about unhandled parameters
        for key in parameters:
            if key not in ['from', 'to', 'limit', 'offset', 'checkids',
                           'contactids', 'status', 'via']:
                sys.stderr.write('%s not a valid argument for actions()\n'
                                 % key)

        response = self.request('GET', 'actions', parameters)

        return response.json()['actions']