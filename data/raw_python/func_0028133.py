def set_maintainer(self, maintainer):
        # type: (Union[hdx.data.user.User,Dict,str]) -> None
        """Set the dataset's maintainer.

         Args:
             maintainer (Union[User,Dict,str]): Either a user id or User metadata from a User object or dictionary.
         Returns:
             None
        """
        if isinstance(maintainer, hdx.data.user.User) or isinstance(maintainer, dict):
            if 'id' not in maintainer:
                maintainer = hdx.data.user.User.read_from_hdx(maintainer['name'], configuration=self.configuration)
            maintainer = maintainer['id']
        elif not isinstance(maintainer, str):
            raise HDXError('Type %s cannot be added as a maintainer!' % type(maintainer).__name__)
        if is_valid_uuid(maintainer) is False:
            raise HDXError('%s is not a valid user id for a maintainer!' % maintainer)
        self.data['maintainer'] = maintainer