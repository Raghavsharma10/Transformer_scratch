def register(self, kind, handler):
		"""Register a handler for a given type, class, interface, or abstract base class.
		
		View registration should happen within the `start` callback of an extension.  For example, to register the
		previous `json` view example:
		
			class JSONExtension:
				def start(self, context):
					context.view.register(tuple, json)
		
		The approach of explicitly referencing a view handler isn't very easy to override without also replacing the
		extension originally adding it, however there is another approach. Using named handlers registered as discrete
		plugins (via the `entry_point` argument in `setup.py`) allows the extension to easily ask "what's my handler?"
		
			class JSONExtension:
				def start(self, context):
					context.view.register(
							tuple,
							context.view.json
						)
		
		Otherwise unknown attributes of the view registry will attempt to look up a handler plugin by that name.
		"""
		if __debug__:  # In production this logging is completely skipped, regardless of logging level.
			if py3 and not pypy:  # Where possible, we shorten things to just the cannonical name.
				log.debug("Registering view handler.", extra=dict(type=name(kind), handler=name(handler)))
			else:  # Canonical name lookup is not entirely reliable on some combinations.
				log.debug("Registering view handler.", extra=dict(type=repr(kind), handler=repr(handler)))
		
		# Add the handler to the pool of candidates. This adds to a list instead of replacing the "dictionary item".
		self._map.add(kind, handler)
		
		return handler