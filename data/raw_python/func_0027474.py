def getContacts(self, **kwargs):
        """Returns a list of all contacts.

        Optional Parameters:

            * limit -- Limits the number of returned contacts to the specified
                quantity.
                    Type: Integer
                    Default: 100

            * offset -- Offset for listing (requires limit.)
                    Type: Integer
                    Default: 0

        Returned structure:
        [
            'id'                 : <Integer> Contact identifier
            'name'               : <String> Contact name
            'email'              : <String> Contact email
            'cellphone'          : <String> Contact telephone
            'countryiso'         : <String> Cellphone country ISO code
            'defaultsmsprovider' : <String> Default SMS provider
            'directtwitter'      : <Boolean> Send Tweets as direct messages
            'twitteruser'        : <String> Twitter username
            'paused'             : <Boolean> True if contact is pasued
            'iphonetokens'       : <String list> iPhone tokens
            'androidtokens'      : <String list> android tokens
        ]
        """

        # Warn user about unhandled parameters
        for key in kwargs:
            if key not in ['limit', 'offset']:
                sys.stderr.write("'%s'" % key + ' is not a valid argument ' +
                                 'of getContacts()\n')

        return [PingdomContact(self, x) for x in
                self.request("GET", "notification_contacts", kwargs).json()['contacts']]