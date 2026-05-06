def Page_deleteCookie(self, cookieName, url):
		"""
		Function path: Page.deleteCookie
			Domain: Page
			Method name: deleteCookie
		
			WARNING: This function is marked 'Experimental'!
		
			Parameters:
				Required arguments:
					'cookieName' (type: string) -> Name of the cookie to remove.
					'url' (type: string) -> URL to match cooke domain and path.
			No return value.
		
			Description: Deletes browser cookie with given name, domain and path.
		"""
		assert isinstance(cookieName, (str,)
		    ), "Argument 'cookieName' must be of type '['str']'. Received type: '%s'" % type(
		    cookieName)
		assert isinstance(url, (str,)
		    ), "Argument 'url' must be of type '['str']'. Received type: '%s'" % type(
		    url)
		subdom_funcs = self.synchronous_command('Page.deleteCookie', cookieName=
		    cookieName, url=url)
		return subdom_funcs