def subscribe(self, observer):
        """Subscribe an observer to this subject and return a subscription id

        """
        sid = self._sn
        self.observers[sid] = observer
        self._sn += 1
        return SubscribeID(self, sid)