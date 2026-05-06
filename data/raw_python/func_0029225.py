def _make_request(self, method, params):
        """Make request to API endpoint

        Note: Ignoring SSL cert validation due to intermittent failures
        http://requests.readthedocs.org/en/latest/user/advanced/#ssl-cert-verification
        """
        if self.blocked_until is not None and \
                datetime.datetime.utcnow() < self.blocked_until:
            raise SlackError("Too many requests - wait until {0}" \
                    .format(self.blocked_until))

        url = "%s/%s" % (SlackClient.BASE_URL, method)
        params['token'] = self.token
        response = requests.post(url, data=params, verify=self.verify)

        if response.status_code == 429:
            # Too many requests
            retry_after = int(response.headers.get('retry-after', '1'))
            self.blocked_until = datetime.datetime.utcnow() + \
                    datetime.timedelta(seconds=retry_after)
            raise SlackError("Too many requests - retry after {0} second(s)" \
                    .format(retry_after))

        result = response.json()
        if not result['ok']:
            raise SlackError(result['error'])
        return result