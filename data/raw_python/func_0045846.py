def _run_namespace(self, doc):
        """Run through all receivers related to the doc's namespace"""
        for receiver in self.receivers[doc['ns']]:
            params = self.receivers[doc['ns']][receiver]
            if params[doc['op']]:
                if params[doc['op']] is True or str(doc['o']['_id']) in params[doc['op']]:
                    notification_id = self._notify_receiver(receiver, params, doc)
                    if params['reliable'] and self._max_attempts > 1:
                        self.notifications[notification_id] = {
                            'receiver': receiver,
                            'params': dict(params),
                            'doc': dict(doc),
                            'ts': datetime.datetime.utcnow(),
                            'attempts': 1}