def get_subscription_labels(self, userPk):
        """Returns a list with all the labels the user is subscribed to"""
        r = self._request('subscriptions/' + str(userPk))
        if r:
            s = r.json()
            return s
        return []