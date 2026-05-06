def delete_account(self):
        """
        Delete a user's account.
        Deleting the user's account can only be done if the user's domain is controlled by the authorized organization,
        and removing the account will also remove the user from all organizations with access to that domain.
        :return: None, because you cannot follow this command with another.
        """
        if self.id_type == IdentityTypes.adobeID:
            raise ArgumentError("You cannot delete an Adobe ID account.")
        self.append(removeFromDomain={})
        return None