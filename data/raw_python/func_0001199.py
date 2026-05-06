def disconnect(self, cback):
        "See signal"
        return self.signal.disconnect(cback,
                                      subscribers=self.subscribers,
                                      instance=self.instance)