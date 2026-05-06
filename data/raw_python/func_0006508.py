def _load_users(self):
        """Default implentation requires users from DB.
        Should setup `users` attribute"""
        r = sql.abstractRequetesSQL.get_users()()
        self.users = {d["id"]: dict(d) for d in r}