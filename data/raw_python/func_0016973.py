def edit(self, config={}, events=[], add_events=[], rm_events=[],
             active=True):
        """Edit this hook.

        :param dict config: (optional), key-value pairs of settings for this
            hook
        :param list events: (optional), which events should this be triggered
            for
        :param list add_events: (optional), events to be added to the list of
           events that this hook triggers for
        :param list rm_events: (optional), events to be remvoed from the list
            of events that this hook triggers for
        :param bool active: (optional), should this event be active
        :returns: bool
        """
        data = {'config': config, 'active': active}
        if events:
            data['events'] = events

        if add_events:
            data['add_events'] = add_events

        if rm_events:
            data['remove_events'] = rm_events

        json = self._json(self._patch(self._api, data=dumps(data)), 200)

        if json:
            self._update_(json)
            return True

        return False