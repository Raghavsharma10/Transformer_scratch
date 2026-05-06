def add_users(self, users=None):
        """
        Add users (specified by email address) to this user group.
        In case of ambiguity (two users with same email address), the non-AdobeID user is preferred.
        :param users: list of emails for users to add to the group.
        :return: the Group, so you can do Group(...).add_users(...).add_to_products(...)
        """
        if not users:
            raise ArgumentError("You must specify emails for users to add to the user group")
        ulist = {"user": [user for user in users]}
        return self.append(add=ulist)