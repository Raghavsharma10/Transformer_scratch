def _handle_login(self, event):
        """Manual password based login"""

        # TODO: Refactor to simplify

        self.log("Auth request for ", event.username, 'client:',
                 event.clientuuid)

        # TODO: Define the requirements for secure passwords etc.
        # They're also required in the Enrol module..!

        if (len(event.username) < 1) or (len(event.password) < 5):
            self.log("Illegal username or password received, login cancelled", lvl=warn)
            self._fail(event, 'Password or username too short')
            return

        client_config = None

        try:
            user_account = objectmodels['user'].find_one({
                'name': event.username
            })
            # self.log("Account: %s" % user_account._fields, lvl=debug)
            if user_account is None:
                raise AuthenticationError
        except Exception as e:
            self.log("No userobject due to error: ", e, type(e),
                     lvl=error)
            self._fail(event)
            return

        self.log("User found.", lvl=debug)

        if user_account.active is False:
            self.log("Account deactivated.")
            self._fail(event, 'Account deactivated.')
            return

        if not std_hash(event.password, self.salt) == user_account.passhash:
            self.log("Password was wrong!", lvl=warn)
            self._fail(event)
            return

        self.log("Passhash matches, checking client and profile.",
                 lvl=debug)

        requested_client_uuid = event.requestedclientuuid
        if requested_client_uuid is not None:
            client_config = objectmodels['client'].find_one({
                'uuid': requested_client_uuid
            })

        if client_config:
            self.log("Checking client configuration permissions",
                     lvl=debug)
            # TODO: Shareable client configurations?
            if client_config.owner != user_account.uuid:
                client_config = None
                self.log("Unauthorized client configuration "
                         "requested",
                         lvl=warn)
        else:
            self.log("Unknown client configuration requested: ",
                     requested_client_uuid, event.__dict__,
                     lvl=warn)

        if not client_config:
            self.log("Creating new default client configuration")
            # Either no configuration was found or not requested
            # -> Create a new client configuration
            uuid = event.clientuuid if event.clientuuid is not None else str(uuid4())

            client_config = objectmodels['client']({'uuid': uuid})

            client_config.name = std_human_uid(kind='place')

            client_config.description = "New client configuration from " + user_account.name
            client_config.owner = user_account.uuid

            # TODO: Get client configuration storage done right, this one is too simple
            client_config.save()

        user_profile = self._get_profile(user_account)

        self._login(event, user_account, user_profile, client_config)
        self.log("Done with Login request", lvl=debug)