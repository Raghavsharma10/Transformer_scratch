def notify(self, *args, **kwargs):
        "See signal"
        loop = kwargs.pop('loop', self.loop)
        return self.signal.prepare_notification(
            subscribers=self.subscribers, instance=self.instance,
            loop=loop).run(*args, **kwargs)