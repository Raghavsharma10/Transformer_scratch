def get_users(self):
        """
        Retrieve all users in the system

        :return:
            A list of :class:`meteorpi_model.User`
        """
        output = []
        self.con.execute('SELECT userId, uid FROM archive_users;')
        results = self.con.fetchall()

        for result in results:
            # Fetch list of roles
            self.con.execute('SELECT name FROM archive_roles r INNER JOIN archive_user_roles u ON u.roleId=r.uid '
                             'WHERE u.userId = %s;', (result['uid'],))
            role_list = [row['name'] for row in self.con.fetchall()]
            output.append(mp.User(user_id=result['userId'], roles=role_list))
        return output