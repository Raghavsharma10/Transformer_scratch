def register_stats_pusher(self, pusher):
        """Registers a pusher to be used for pushing statistics to various remotes/locals.

        :param Pusher|list[Pusher] pusher:

        """
        for pusher in listify(pusher):
            self._set('stats-push', pusher, multi=True)

        return self._section