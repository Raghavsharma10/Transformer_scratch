def connect(self, cback):
        "See signal"
        return self.signal.connect(cback,
                                   subscribers=self.subscribers,
                                   instance=self.instance)