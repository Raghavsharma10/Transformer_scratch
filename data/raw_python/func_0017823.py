def create_admin_set_password(self, password):
        """
        create 'admin' account with given password
        """
        with open(self.sitedir + '/run/admin.json', 'w') as out:
            json.dump({
                'name': 'admin',
                'email': 'none',
                'password': password,
                'sysadmin': True},
                out)
        self.user_run_script(
            script=scripts.get_script_path('update_add_admin.sh'),
            args=[],
            db_links=True,
            ro={
                self.sitedir + '/run/admin.json': '/input/admin.json'
               },
            )
        remove(self.sitedir + '/run/admin.json')