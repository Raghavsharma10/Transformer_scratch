def loggin(self, user_id, mdp, autolog):
        """Check mdp and return True it's ok"""
        r = sql.abstractRequetesSQL.check_mdp_user(user_id, mdp)
        if r():
            # update auto-log params
            self.autolog[user_id] = autolog and mdp or False
            self.modules = self.users[user_id]["modules"]  # load modules list

            dic = {"autolog": self.autolog, "modules": self.modules}
            s = json.dumps(dic, indent=4, ensure_ascii=False)
            b = security.protege_data(s, True)
            with open("local/init", "wb") as f:
                f.write(b)

            self.mode_online = True  # authorization to execute bakground tasks
            return True
        else:
            logging.debug("Bad password !")