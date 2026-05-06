def channels_list(self, exclude_archived=True, **params):
        """channels.list

        This method returns a list of all channels in the team. This includes
        channels the caller is in, channels they are not currently in, and
        archived channels. The number of (non-deactivated) members in each
        channel is also returned.

        https://api.slack.com/methods/channels.list
        """
        method = 'channels.list'
        params.update({'exclude_archived': exclude_archived and 1 or 0})
        return self._make_request(method, params)