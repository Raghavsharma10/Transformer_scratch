def _send_request(self, url, needs_auth=False, **kwargs):
        """
        Handles top level functionality for sending requests to Imgur.

        This mean
            - Raising client-side error if insufficient authentication.
            - Adding authentication information to the request.
            - Split the request into multiple request for pagination.
            - Retry calls for certain server-side errors.
            - Refresh access token automatically if expired.
            - Updating ratelimit info

        :param needs_auth: Is authentication as a user needed for the execution
            of this method?
        """
        # TODO: Add automatic test for timed_out access_tokens and
        # automatically refresh it before carrying out the request.
        if self.access_token is None and needs_auth:
            # TODO: Use inspect to insert name of method in error msg.
            raise Exception("Authentication as a user is required to use this "
                            "method.")
        if self.access_token is None:
            # Not authenticated as a user. Use anonymous access.
            auth = {'Authorization': 'Client-ID {0}'.format(self.client_id)}
        else:
            auth = {'Authorization': 'Bearer {0}'.format(self.access_token)}
        if self.mashape_key:
            auth.update({'X-Mashape-Key': self.mashape_key})
        content = []
        is_paginated = False
        if 'limit' in kwargs:
            is_paginated = True
            limit = kwargs['limit'] or self.DEFAULT_LIMIT
            del kwargs['limit']
            page = 0
            base_url = url
            url.format(page)
        kwargs['authentication'] = auth
        while True:
            result = request.send_request(url, verify=self.verify, **kwargs)
            new_content, ratelimit_info = result
            if is_paginated and new_content and limit > len(new_content):
                content += new_content
                page += 1
                url = base_url.format(page)
            else:
                if is_paginated:
                    content = (content + new_content)[:limit]
                else:
                    content = new_content
                break
        # Note: When the cache is implemented, it's important that the
        # ratelimit info doesn't get updated with the ratelimit info in the
        # cache since that's likely incorrect.
        for key, value in ratelimit_info.items():
            setattr(self, key[2:].replace('-', '_'), value)
        return content