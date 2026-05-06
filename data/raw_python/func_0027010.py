def process(self, event):
        """ Send events as push notification via Google Cloud Messaging.
            Expected settings as follows:

                # https://developers.google.com/mobile/add
                WALDUR_CORE['GOOGLE_API'] = {
                    'NOTIFICATION_TITLE': "Waldur notification",
                    'Android': {
                        'server_key': 'AIzaSyA2_7UaVIxXfKeFvxTjQNZbrzkXG9OTCkg',
                    },
                    'iOS': {
                        'server_key': 'AIzaSyA34zlG_y5uHOe2FmcJKwfk2vG-3RW05vk',
                    }
                }
        """

        conf = settings.WALDUR_CORE.get('GOOGLE_API') or {}
        keys = conf.get(dict(self.Type.CHOICES)[self.type])

        if not keys or not self.token:
            return

        endpoint = 'https://gcm-http.googleapis.com/gcm/send'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': 'key=%s' % keys['server_key'],
        }
        payload = {
            'to': self.token,
            'notification': {
                'body': event.get('message', 'New event'),
                'title': conf.get('NOTIFICATION_TITLE', 'Waldur notification'),
                'image': 'icon',
            },
            'data': {
                'event': event
            },
        }
        if self.type == self.Type.IOS:
            payload['content-available'] = '1'
        logger.debug('Submitting GCM push notification with headers %s, payload: %s' % (headers, payload))
        requests.post(endpoint, json=payload, headers=headers)