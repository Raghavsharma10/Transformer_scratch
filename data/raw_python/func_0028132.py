def get_maintainer(self):
        # type: () -> hdx.data.user.User
        """Get the dataset's maintainer.

         Returns:
             User: Dataset's maintainer
        """
        return hdx.data.user.User.read_from_hdx(self.data['maintainer'], configuration=self.configuration)