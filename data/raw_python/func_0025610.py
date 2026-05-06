def after(self, context, exc=None):
		"""Executed after dispatch has returned and the response populated, prior to anything being sent to the client."""
		
		duration = context._duration = round((time.time() - context._start_time) * 1000)  # Convert to ms.
		delta = unicode(duration)
		
		# Default response augmentation.
		if self.header:
			context.response.headers[self.header] = delta
		
		if self.log:
			self.log("Response generated in " + delta + " seconds.", extra=dict(
					duration = duration,
					request = id(context)
				))