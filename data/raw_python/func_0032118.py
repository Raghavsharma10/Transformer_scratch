def installOrResume(self, userstore):
        """
        Install this product on a user store. If this product has been
        installed on the user store already and the installation is suspended,
        it will be resumed. If it exists and is not suspended, an error will be
        raised.
        """
        for i in userstore.query(Installation, Installation.types == self.types):
            if i.suspended:
                unsuspendTabProviders(i)
                return
            else:
                raise RuntimeError("installOrResume called for an"
                                   " installation that isn't suspended")
        else:
            self.installProductOn(userstore)