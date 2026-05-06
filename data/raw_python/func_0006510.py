def has_autolog(self, user_id):
        """
        Read auto-connection parameters and returns local password or None
        """
        try:
            with open("local/init", "rb") as f:
                s = f.read()
                s = security.protege_data(s, False)
                self.autolog = json.loads(s).get("autolog", {})
        except FileNotFoundError:
            return

        mdp = self.autolog.get(user_id, None)
        return mdp