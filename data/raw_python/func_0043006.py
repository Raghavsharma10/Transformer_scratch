def create_or_update_user(self, user_id, password, roles):
        """
        Create a new user record, or update an existing one

        :param user_id:
            user ID to update or create
        :param password:
            new password, or None to leave unchanged
        :param roles:
            new roles, or None to leave unchanged
        :return:
            the action taken, one of "none", "update", "create"
        :raises:
            ValueError if there is no existing user and either password or roles is None
        """
        action = "update"
        self.con.execute('SELECT 1 FROM archive_users WHERE userId = %s;', (user_id,))
        results = self.con.fetchall()
        if len(results) == 0:
            if password is None:
                raise ValueError("Must specify an initial password when creating a new user!")
            action = "create"
            self.con.execute('INSERT INTO archive_users (userId, pwHash) VALUES (%s,%s)',
                             (user_id, passlib.hash.bcrypt.encrypt(password)))

        if password is None and roles is None:
            action = "none"
        if password is not None:
            self.con.execute('UPDATE archive_users SET pwHash = %s WHERE userId = %s',
                             (passlib.hash.bcrypt.encrypt(password), user_id))
        if roles is not None:

            # Clear out existing roles, and delete any unused roles
            self.con.execute("DELETE r FROM archive_user_roles AS r WHERE "
                             "(SELECT u.userId FROM  archive_users AS u WHERE r.userId=u.uid)=%s;", (user_id,))
            self.con.execute("DELETE r FROM archive_roles AS r WHERE r.uid NOT IN "
                             "(SELECT roleId FROM archive_user_roles);")

            for role in roles:
                self.con.execute("SELECT uid FROM archive_roles WHERE name=%s;", (role,))
                results = self.con.fetchall()
                if len(results) < 1:
                    self.con.execute("INSERT INTO archive_roles (name) VALUES (%s);", (role,))
                    self.con.execute("SELECT uid FROM archive_roles WHERE name=%s;", (role,))
                    results = self.con.fetchall()

                self.con.execute('INSERT INTO archive_user_roles (userId, roleId) VALUES '
                                 '((SELECT u.uid FROM archive_users u WHERE u.userId=%s),'
                                 '%s)', (user_id, results[0]['uid']))
            return action