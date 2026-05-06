def destroy(self, bot_id):
        """Destroy a bot.

        :param str bot_id: the ID of the bot to destroy
        :return: ``True`` if successful
        :rtype: bool
        """
        url = utils.urljoin(self.url, 'destroy')
        payload = {'bot_id': bot_id}
        response = self.session.post(url, json=payload)
        return response.ok