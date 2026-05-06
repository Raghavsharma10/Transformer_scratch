def close_chromium(self):
		'''
		Close the remote chromium instance.

		This command is normally executed as part of the class destructor.
		It can be called early without issue, but calling ANY class functions
		after the remote chromium instance is shut down will have unknown effects.

		Note that if you are rapidly creating and destroying ChromeController instances,
		you may need to *explicitly* call this before destruction.
		'''
		if self.cr_proc:
			try:
				if 'win' in sys.platform:
					self.__close_internal_windows()
				else:
					self.__close_internal_linux()
			except Exception as e:
				for line in traceback.format_exc().split("\n"):
					self.log.error(line)



		ACTIVE_PORTS.discard(self.port)