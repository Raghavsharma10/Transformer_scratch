def update(self, email=None, username=None, first_name=None, last_name=None, country=None):
        """
        Update values on an existing user.  See the API docs for what kinds of update are possible.
        :param email: new email for this user
        :param username: new username for this user
        :param first_name: new first name for this user
        :param last_name: new last name for this user
        :param country: new country for this user
        :return: the User, so you can do User(...).update(...).add_to_groups(...)
        """
        if username and self.id_type != IdentityTypes.federatedID:
            raise ArgumentError("You cannot set username except for a federated ID")
        if username and '@' in username and not email:
            raise ArgumentError("Cannot update email-type username when email is not specified")
        if email and username and email.lower() == username.lower():
            raise ArgumentError("Specify just email to set both email and username for a federated ID")
        updates = {}
        for k, v in six.iteritems(dict(email=email, username=username,
                                       firstname=first_name, lastname=last_name,
                                       country=country)):
            if v:
                updates[k] = v
        return self.append(update=updates)