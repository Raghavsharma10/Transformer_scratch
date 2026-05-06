def wrap_json(cls, json):
        """Create a User instance for the given json

        :param json: the dict with the information of the user
        :type json: :class:`dict` | None
        :returns: the new user instance
        :rtype: :class:`User`
        :raises: None
        """
        u = User(usertype=json['type'],
                 name=json['name'],
                 logo=json['logo'],
                 twitchid=json['_id'],
                 displayname=json['display_name'],
                 bio=json['bio'])
        return u