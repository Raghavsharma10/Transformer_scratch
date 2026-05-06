def serve(self, service='auto', **options):  # pragma: no cover
		"""Initiate a web server service to serve this application.
		
		You can always use the Application instance as a bare WSGI application, of course.  This method is provided as
		a convienence.
		
		Pass in the name of the service you wish to use, and any additional configuration options appropriate for that
		service.  Almost all services accept `host` and `port` options, some also allow you to specify an on-disk
		`socket`.  By default all web servers will listen to `127.0.0.1` (loopback only) on port 8080.
		"""
		
		service = load(service, 'web.server')  # We don't bother with a full registry for these one-time lookups.
		
		try:
			service(self, **options)
		except KeyboardInterrupt:  # We catch this as SIG_TERM or ^C are basically the only ways to stop most servers.
			pass
		
		# Notify extensions that the service has returned and we are exiting.
		for ext in self.__context.extension.signal.stop: ext(self.__context)