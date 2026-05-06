def wrap_json(cls, json, viewers=None, channels=None):
        """Create a Game instance for the given json

        :param json: the dict with the information of the game
        :type json: :class:`dict`
        :param viewers: The viewer count
        :type viewers: :class:`int`
        :param channels: The viewer count
        :type channels: :class:`int`
        :returns: the new game instance
        :rtype: :class:`Game`
        :raises: None
        """
        g = Game(name=json.get('name'),
                 box=json.get('box'),
                 logo=json.get('logo'),
                 twitchid=json.get('_id'),
                 viewers=viewers,
                 channels=channels)
        return g