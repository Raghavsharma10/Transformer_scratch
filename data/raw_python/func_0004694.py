def reply(self, username, type='text', sender=None, **kwargs):
        """Create the reply text for weixin.

        The reply varies per reply type. The acceptable types are `text`,
        `music`, `news`, `image`, `voice`, `video`. Each type accepts
        different parameters, but they share some common parameters:

            * username: the receiver's username
            * type: the reply type, aka text, music and news
            * sender: sender is optional if you have a default value

        Text reply requires an additional parameter of `content`.

        Music reply requires 4 more parameters:

            * title: A string for music title
            * description: A string for music description
            * music_url: A link of the music
            * hq_music_url: A link of the high quality music

        News reply requires an additional parameter of `articles`, which
        is a list/tuple of articles, each one is a dict:

            * title: A string for article title
            * description: A string for article description
            * picurl: A link for article cover image
            * url: A link for article url

        Image and Voice reply requires an additional parameter of `media_id`.

        Video reply requires 3 more parameters:

            * media_id: A string for video `media_id`
            * title: A string for video title
            * description: A string for video description
        """
        sender = sender or self.sender
        if not sender:
            raise RuntimeError('WEIXIN_SENDER or sender argument is missing')

        if type == 'text':
            content = kwargs.get('content', '')
            return text_reply(username, sender, content)

        if type == 'music':
            values = {}
            for k in ('title', 'description', 'music_url', 'hq_music_url'):
                values[k] = kwargs.get(k)
            return music_reply(username, sender, **values)

        if type == 'news':
            items = kwargs.get('articles', [])
            return news_reply(username, sender, *items)

        if type == 'customer_service':
            service_account = kwargs.get('service_account', None)
            return transfer_customer_service_reply(username, sender,
                                                   service_account)

        if type == 'image':
            media_id = kwargs.get('media_id', '')
            return image_reply(username, sender, media_id)

        if type == 'voice':
            media_id = kwargs.get('media_id', '')
            return voice_reply(username, sender, media_id)

        if type == 'video':
            values = {}
            for k in ('media_id', 'title', 'description'):
                values[k] = kwargs.get(k)
            return video_reply(username, sender, **values)