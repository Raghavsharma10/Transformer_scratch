def get_stats(self, start=int(time()), stop=int(time())+10, step=10):
        """
        Get stats of a monitored machine

        :param start: Time formatted as integer, from when to fetch stats (default now)
        :param stop: Time formatted as integer, until when to fetch stats (default +10 seconds)
        :param step: Step to fetch stats (default 10 seconds)
        :returns: A dict of stats
        """
        payload = {
            'v': 2,
            'start': start,
            'stop': stop,
            'step': step
        }

        data = json.dumps(payload)
        req = self.request(self.mist_client.uri+"/clouds/"+self.cloud.id+"/machines/"+self.id+"/stats", data=data)
        stats = req.get().json()
        return stats