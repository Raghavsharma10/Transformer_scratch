def run(self):
        """
        Creates user, encrypts password.
        """
        from zengine.models import User
        user = User(username=self.manager.args.username, superuser=self.manager.args.super)
        user.set_password(self.manager.args.password)
        user.save()
        print("New user created with ID: %s" % user.key)