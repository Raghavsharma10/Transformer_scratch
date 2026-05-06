def wrap_json(cls, json):
        """Create a Stream instance for the given json

        :param json: the dict with the information of the stream
        :type json: :class:`dict` | None
        :returns: the new stream instance
        :rtype: :class:`Stream` | None
        :raises: None
        """
        if json is None:
            return None
        channel = Channel.wrap_json(json.get('channel'))
        s = Stream(game=json.get('game'),
                   channel=channel,
                   twitchid=json.get('_id'),
                   viewers=json.get('viewers'),
                   preview=json.get('preview'))
        return s