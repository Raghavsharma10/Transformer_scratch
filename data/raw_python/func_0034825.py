def update_handler(feeds):
		'''Update all cross-referencing filters results for feeds and others, related to them.
			Intended to be called from non-Feed update hooks (like new Post saving).'''
		# Check if this call is a result of actions initiated from
		#  one of the hooks in a higher frame (resulting in recursion).
		if Feed._filters_update_handler_lock: return
		return Feed._filters_update_handler(Feed, feeds, force=True)