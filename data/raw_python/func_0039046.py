def __pre_check(self, requestedUrl):
		'''
		Allow the pre-emptive fetching of sites with a full browser if they're known
		to be dick hosters.
		'''
		components = urllib.parse.urlsplit(requestedUrl)

		netloc_l = components.netloc.lower()
		if netloc_l in Domain_Constants.SUCURI_GARBAGE_SITE_NETLOCS:
			self.__check_suc_cookie(components)
		elif netloc_l in Domain_Constants.CF_GARBAGE_SITE_NETLOCS:
			self.__check_cf_cookie(components)
		elif components.path == '/sucuri_shit_2':
			self.__check_suc_cookie(components)
		elif components.path == '/sucuri_shit_3':
			self.__check_suc_cookie(components)
		elif components.path == '/cloudflare_under_attack_shit_2':
			self.__check_cf_cookie(components)
		elif components.path == '/cloudflare_under_attack_shit_3':
			self.__check_cf_cookie(components)