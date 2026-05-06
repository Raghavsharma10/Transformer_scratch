def newContact(self, name, **kwargs):
        """Create a new contact.

        Provide new contact name and any optional arguments. Returns new
            PingdomContact instance

        Optional Parameters:

            * email -- Contact email address
                    Type: String

            * cellphone -- Cellphone number, without the country code part. In
                some countries you are supposed to exclude leading zeroes.
                (Requires countrycode and countryiso)
                    Type: String

            * countrycode -- Cellphone country code (Requires cellphone and
                countryiso)
                    Type: String

            * countryiso -- Cellphone country ISO code. For example: US (USA),
                GB (Britain) or SE (Sweden) (Requires cellphone and
                countrycode)
                    Type: String

            * defaultsmsprovider -- Default SMS provider
                    Type: String ['clickatell', 'bulksms', 'esendex',
                                  'cellsynt']

            * directtwitter -- Send tweets as direct messages
                    Type: Boolean
                    Default: True

            * twitteruser -- Twitter user
                    Type: String
        """

        # Warn user about unhandled parameters
        for key in kwargs:
            if key not in ['email', 'cellphone', 'countrycode', 'countryiso',
                           'defaultsmsprovider', 'directtwitter',
                           'twitteruser']:
                sys.stderr.write("'%s'" % key + ' is not a valid argument ' +
                                 'of newContact()\n')

        kwargs['name'] = name
        contactinfo = self.request("POST", "notification_contacts",
                                   kwargs).json()['contact']

        return PingdomContact(self, contactinfo)