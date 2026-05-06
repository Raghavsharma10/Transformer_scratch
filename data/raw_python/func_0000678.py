def list(self, term=None):
        """List alerts configured for the account."""
        if not self._state:
            raise InvalidState("State was not properly obtained from the app")
        self._process_state()
        if not self._state[1]:
            self._log.info("No monitors have been created yet.")
            return list()

        monitors = list()
        for monitor in self._state[1][1]:
            obj = dict()
            obj['monitor_id'] = monitor[1]
            obj['user_id'] = monitor[-1]
            obj['term'] = monitor[2][3][1]
            if term and obj['term'] != term:
                continue
            obj['language'] = monitor[2][3][3][1]
            obj['region'] = monitor[2][3][3][2]
            obj['delivery'] = self.DELIVERY[monitor[2][6][0][1]]
            obj['match_type'] = self.MONITOR_MATCH_TYPE[monitor[2][5]]
            if obj['delivery'] == 'MAIL':
                obj['alert_frequency'] = self.ALERT_FREQ[monitor[2][6][0][4]]
                obj['email_address'] = monitor[2][6][0][2]
            else:
                rss_id = monitor[2][6][0][11]
                url = "https://google.com/alerts/feeds/{uid}/{fid}"
                obj['rss_link'] = url.format(uid=obj['user_id'], fid=rss_id)
            monitors.append(obj)
        return monitors