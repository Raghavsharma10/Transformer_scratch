def get_stats(self, service_id, stat_type=FastlyStatsType.ALL):
		"""Get the stats from a service."""
		content = self._fetch("/service/%s/stats/%s" % (service_id, stat_type))
		return content