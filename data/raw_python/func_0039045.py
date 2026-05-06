def __check_suc_cookie(self, components):
		'''
		This is only called if we're on a known sucuri-"protected" site.
		As such, if we do *not* have a sucuri cloudproxy cookie, we can assume we need to
		do the normal WAF step-through.
		'''
		netloc = components.netloc.lower()

		for cookie in self.cj:
			if cookie.domain_specified and (cookie.domain.lower().endswith(netloc)
				or (cookie.domain.lower().endswith("127.0.0.1") and (
				components.path == "/sucuri_shit_3" or components.path == "/sucuri_shit_2" ))):   # Allow testing
				if "sucuri_cloudproxy_uuid_" in cookie.name:
					return
		self.log.info("Missing cloudproxy cookie for known sucuri wrapped site. Doing a pre-emptive chromium fetch.")
		raise Exceptions.SucuriWrapper("WAF Shit", str(components))