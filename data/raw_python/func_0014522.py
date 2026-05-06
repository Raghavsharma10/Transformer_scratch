def set_password(self, admin, new_password, old_password):
        """Set an admin's password.

        :param admin: Name of admin whose password is to be set.
        :type admin: str
        :param new_password: New password for admin.
        :type new_password: str
        :param old_password: Current password of admin.
        :type old_password: str

        :returns: A dictionary mapping "name" to admin.
        :rtype: ResponseDict

        """
        return self.set_admin(admin, password=new_password,
                              old_password=old_password)