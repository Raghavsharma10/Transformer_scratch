def list_queue(self, embed_last_unused_offers=False):
        """List all the tasks queued up or waiting to be scheduled.

        :returns: list of queue items
        :rtype: list[:class:`marathon.models.queue.MarathonQueueItem`]
        """
        if embed_last_unused_offers:
            params = {'embed': 'lastUnusedOffers'}
        else:
            params = {}
        response = self._do_request('GET', '/v2/queue', params=params)
        return self._parse_response(response, MarathonQueueItem, is_list=True, resource_name='queue')