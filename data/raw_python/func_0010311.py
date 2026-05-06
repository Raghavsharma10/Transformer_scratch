def get_monitor(self, topics):
        """Attempts to find a Monitor in device cloud that matches the provided topics

        :param topics: a string list of topics (e.g. ``['DeviceCore[U]', 'FileDataCore'])``)

        Returns a :class:`DeviceCloudMonitor` if found, otherwise None.
        """
        for monitor in self.get_monitors(MON_TOPIC_ATTR == ",".join(topics)):
            return monitor  # return the first one, even if there are multiple
        return None