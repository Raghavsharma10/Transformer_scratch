def remove_users(self, users=None):
        """
        Remove users (specified by email address) from this user group.
        In case of ambiguity (two users with same email address), the non-AdobeID user is preferred.
        :param users: list of emails for users to remove from the group.
        :return: the Group, so you can do Group(...).remove_users(...).add_to_products(...)
        """
        if not users:
            raise ArgumentError("You must specify emails for users to remove from the user group")
        ulist = {"user": [user for user in users]}
        return self.append(remove=ulist)