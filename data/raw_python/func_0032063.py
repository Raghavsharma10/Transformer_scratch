def userCreate(self, request, tag):
        """
        Render a form for creating new users.
        """
        userCreator = liveform.LiveForm(
            self.createUser,
            [liveform.Parameter(
                    "localpart",
                    liveform.TEXT_INPUT,
                    unicode,
                    "localpart"),
             liveform.Parameter(
                    "domain",
                    liveform.TEXT_INPUT,
                    unicode,
                    "domain"),
             liveform.Parameter(
                    "password",
                    liveform.PASSWORD_INPUT,
                    unicode,
                    "password")])
        userCreator.setFragmentParent(self)
        return userCreator