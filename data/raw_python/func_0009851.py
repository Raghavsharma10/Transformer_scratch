def Page_getResourceContent(self, frameId, url):
		"""
		Function path: Page.getResourceContent
			Domain: Page
			Method name: getResourceContent
		
			WARNING: This function is marked 'Experimental'!
		
			Parameters:
				Required arguments:
					'frameId' (type: FrameId) -> Frame id to get resource for.
					'url' (type: string) -> URL of the resource to get content for.
			Returns:
				'content' (type: string) -> Resource content.
				'base64Encoded' (type: boolean) -> True, if content was served as base64.
		
			Description: Returns content of the given resource.
		"""
		assert isinstance(url, (str,)
		    ), "Argument 'url' must be of type '['str']'. Received type: '%s'" % type(
		    url)
		subdom_funcs = self.synchronous_command('Page.getResourceContent',
		    frameId=frameId, url=url)
		return subdom_funcs