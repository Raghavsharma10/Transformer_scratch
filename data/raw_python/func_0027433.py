def get_user_and_check_auth(self, username, password):
        """Check the combination username/password that is valid on the
        database.
        """
        constraint = sql.or_(
            models.USERS.c.name == username,
            models.USERS.c.email == username
        )

        user = self.identity_from_db(models.USERS, constraint)
        if user is None:
            raise dci_exc.DCIException('User %s does not exists.' % username,
                                       status_code=401)

        return user, auth.check_passwords_equal(password, user.password)