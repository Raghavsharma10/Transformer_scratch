def events(self, event_id):
        """
        This method is primarily used to report on the progress of an event
        by providing the percentage of completion.

        Required parameters

            event_id:
                Numeric, this is the id of the event you would like more
                information about
        """
        json = self.request('/events/%s' % event_id, method='GET')
        status = json.get('status')
        if status == 'OK':
            event_json = json.get('event')
            event = Event.from_json(event_json)
            return event
        else:
            message = json.get('message')
            raise DOPException('[%s]: %s' % (status, message))