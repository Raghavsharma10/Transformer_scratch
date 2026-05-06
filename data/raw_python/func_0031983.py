def wrap_json(cls, json):
        """Create a Channel instance for the given json

        :param json: the dict with the information of the channel
        :type json: :class:`dict`
        :returns: the new channel instance
        :rtype: :class:`Channel`
        :raises: None
        """
        c = Channel(name=json.get('name'),
                    status=json.get('status'),
                    displayname=json.get('display_name'),
                    game=json.get('game'),
                    twitchid=json.get('_id'),
                    views=json.get('views'),
                    followers=json.get('followers'),
                    url=json.get('url'),
                    language=json.get('language'),
                    broadcaster_language=json.get('broadcaster_language'),
                    mature=json.get('mature'),
                    logo=json.get('logo'),
                    banner=json.get('banner'),
                    video_banner=json.get('video_banner'),
                    delay=json.get('delay'))
        return c