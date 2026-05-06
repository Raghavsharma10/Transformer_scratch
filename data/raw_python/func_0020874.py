def save_alert(self, email, type, template, when=None, options=None):
        """
        Add a new alert to a user. You can add either a realtime or a summary alert (daily/weekly).
        http://docs.sailthru.com/api/alert

        Usage:
            email = 'praj@sailthru.com'
            type = 'weekly'
            template = 'default'
            when = '+5 hours'
            alert_options = {'match': {}, 'min': {}, 'max': {}, 'tags': []}
            alert_options['match']['type'] = 'shoes'
            alert_options['min']['price'] = 20000 #cents
            alert_options['tags'] = ['red', 'blue', 'green']
            response = client.save_alert(email, type, template, when, alert_options)

        @param email: Email value
        @param type: daily|weekly|realtime
        @param template: template name
        @param when: date string required for summary alert (daily/weekly)
        @param options: dictionary value for adding tags, max price, min price, match type
        """
        options = options or {}
        data = options.copy()
        data['email'] = email
        data['type'] = type
        data['template'] = template
        if type in ['weekly', 'daily']:
            data['when'] = when
        return self.api_post('alert', data)