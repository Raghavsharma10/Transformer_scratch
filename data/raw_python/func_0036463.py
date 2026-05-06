def get_company_user(self, email):
        """Get company user based on email.

        :param email: address of contact
        :type email: ``str``, ``unicode``
        :rtype: ``dict`` with contact information
        """

        users = self.get_company_users()
        for user in users:
            if user['email'] == email:
                return user

        msg = 'No user with email: "{email}" associated with this company.'
        raise FMBaseError(msg.format(email=email))