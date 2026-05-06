def calculate_check_interval( self,
			max_interval, ewma_factor, max_days=None,
			max_updates=None, ewma=0, ewma_ts=None,
			add_partial=None ):
		'''Calculate interval for checks as average
			time (ewma) between updates for specified period.'''
		if not add_partial:
			posts_base = self.posts.only('date_modified').order_by('date_modified')
			if ewma_ts: posts_base = posts_base.filter(date_modified__gt=ewma_ts)
			posts = posts_base
			if max_days:
				posts = posts.filter(date_modified__gt=timezone.now() - timedelta(max_days))
			if max_updates and max_updates > 0:
				posts = posts[:max_updates]
				if len(posts) < max_updates: posts = posts_base[:max_updates]
			timestamps = posts.values_list('date_modified', flat=True)
		else: timestamps = list()
		if add_partial:
			if not ewma_ts:
				try:
					ewma_ts = self.posts.only('date_modified')\
						.order_by('-date_modified')[0].date_modified
				except (ObjectDoesNotExist, IndexError): return 0 # no previous timestamp available
			timestamps.append(add_partial)
			if (add_partial - ewma_ts).total_seconds() < ewma:
				# It doesn't make sense to lower interval due to frequent check attempts.
				return ewma
		for ts in timestamps:
			if ewma_ts is None: # first post
				ewma_ts = ts
				continue
			ewma_ts, interval = ts, (ts - ewma_ts).total_seconds()
			ewma = ewma_factor * interval + (1 - ewma_factor) * ewma
		return min(timedelta(max_interval).total_seconds(), ewma)