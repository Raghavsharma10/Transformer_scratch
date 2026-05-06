def remove_from_organization(self, delete_account=False):
        """
        Remove a user from the organization's list of visible users.  Optionally also delete the account.
        Deleting the account can only be done if the organization owns the account's domain.
        :param delete_account: Whether to delete the account after removing from the organization (default false)
        :return: None, because you cannot follow this command with another.
        """
        self.append(removeFromOrg={"deleteAccount": True if delete_account else False})
        return None