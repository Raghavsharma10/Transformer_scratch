def prepare(self, context):
		"""Add the usual suspects to the context.
		
		This adds `request`, `response`, and `path` to the `RequestContext` instance.
		"""
		
		if __debug__:
			log.debug("Preparing request context.", extra=dict(request=id(context)))
		
		# Bridge in WebOb `Request` and `Response` objects.
		# Extensions shouldn't rely on these, using `environ` where possible instead.
		context.request = Request(context.environ)
		context.response = Response(request=context.request)
		
		# Record the initial path representing the point where a front-end web server bridged to us.
		context.environ['web.base'] = context.request.script_name
		
		# Track the remaining (unprocessed) path elements.
		context.request.remainder = context.request.path_info.split('/')
		if context.request.remainder and not context.request.remainder[0]:
			del context.request.remainder[0]
		
		# Track the "breadcrumb list" of dispatch through distinct controllers.
		context.path = Bread()