def create(self, first_name=None, last_name=None, country=None, email=None,
               on_conflict=IfAlreadyExistsOptions.ignoreIfAlreadyExists):
        """
        Create the user on the Adobe back end.
        See [Issue 32](https://github.com/adobe-apiplatform/umapi-client.py/issues/32): because
        we retry create calls if they time out, the default conflict handling for creation is to ignore the
        create call if the user already exists (possibly from an earlier call that timed out).
        :param first_name: (optional) user first name
        :param last_name: (optional) user last name
        :param country: (optional except for Federated ID) user 2-letter ISO country code
        :param email: user email, if not already specified at create time
        :param on_conflict: IfAlreadyExistsOption (or string name thereof) controlling creation of existing users
        :return: the User, so you can do User(...).create(...).add_to_groups(...)
        """
        # first validate the params: email, on_conflict, first_name, last_name, country
        create_params = {}
        if email is None:
            if not self.email:
                raise ArgumentError("You must specify email when creating a user")
        elif self.email is None:
            self.email = email
        elif self.email.lower() != email.lower():
            raise ArgumentError("Specified email (%s) doesn't match user's email (%s)" % (email, self.email))
        create_params["email"] = self.email
        if on_conflict in IfAlreadyExistsOptions.__members__:
            on_conflict = IfAlreadyExistsOptions[on_conflict]
        if on_conflict not in IfAlreadyExistsOptions:
            raise ArgumentError("on_conflict must be one of {}".format([o.name for o in IfAlreadyExistsOptions]))
        if on_conflict != IfAlreadyExistsOptions.errorIfAlreadyExists:
            create_params["option"] = on_conflict.name
        if first_name: create_params["firstname"] = first_name  # per issue #54 now allowed for all identity types
        if last_name: create_params["lastname"] = last_name     # per issue #54 now allowed for all identity types
        if country: create_params["country"] = country          # per issue #55 should not be defaulted

        # each type is created using a different call
        if self.id_type == IdentityTypes.adobeID:
            return self.insert(addAdobeID=dict(**create_params))
        elif self.id_type == IdentityTypes.enterpriseID:
            return self.insert(createEnterpriseID=dict(**create_params))
        else:
            return self.insert(createFederatedID=dict(**create_params))