def get_emote_picture(self, emote, size=1.0):
        """Return the picture for the given emote

        :param emote: the emote object
        :type emote: :class:`pytwitcherapi.chat.message.Emote`
        :param size: the size of the picture.
                     Choices are: 1.0, 2.0, 3.0
        :type size: :class:`float`
        :returns: A string resembling the picturedata of the emote
        :rtype: :class:`str`
        :raises: None
        """
        r = self.get('http://static-cdn.jtvnw.net/emoticons/v1/%s/%s' %
                     (emote.emoteid, size))
        return r.content