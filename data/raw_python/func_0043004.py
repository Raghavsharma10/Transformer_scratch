def get_user(self, user_id, password):
        """
        Retrieve a user record

        :param user_id:
            the user ID
        :param password:
            password
        :return:
            A :class:`meteorpi_model.User` if everything is correct
        :raises:
            ValueError if the user is found but password is incorrect or if the user is not found.
        """
        self.con.execute('SELECT uid, pwHash FROM archive_users WHERE userId = %s;', (user_id,))
        results = self.con.fetchall()
        if len(results) == 0:
            raise ValueError("No such user")
        pw_hash = results[0]['pwHash']
        # Check the password
        if not passlib.hash.bcrypt.verify(password, pw_hash):
            raise ValueError("Incorrect password")

        # Fetch list of roles
        self.con.execute('SELECT name FROM archive_roles r INNER JOIN archive_user_roles u ON u.roleId=r.uid '
                         'WHERE u.userId = %s;', (results[0]['uid'],))
        role_list = [row['name'] for row in self.con.fetchall()]
        return mp.User(user_id=user_id, roles=role_list)