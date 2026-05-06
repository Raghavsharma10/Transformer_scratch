def social_media(username, platform='twitter', size='medium'):
        """Return avatar URL at social media.
        Visit https://avatars.io for more information.

        :param username: The username of the social media.
        :param platform: One of facebook, instagram, twitter, gravatar.
        :param size: The size of avatar, one of small, medium and large.
        """
        return 'https://avatars.io/{platform}/{username}/{size}'.format(
            platform=platform, username=username, size=size)