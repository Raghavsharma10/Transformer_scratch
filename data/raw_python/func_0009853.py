def Page_setDocumentContent(self, frameId, html):
		"""
		Function path: Page.setDocumentContent
			Domain: Page
			Method name: setDocumentContent
		
			WARNING: This function is marked 'Experimental'!
		
			Parameters:
				Required arguments:
					'frameId' (type: FrameId) -> Frame id to set HTML for.
					'html' (type: string) -> HTML content to set.
			No return value.
		
			Description: Sets given markup as the document's HTML.
		"""
		assert isinstance(html, (str,)
		    ), "Argument 'html' must be of type '['str']'. Received type: '%s'" % type(
		    html)
		subdom_funcs = self.synchronous_command('Page.setDocumentContent',
		    frameId=frameId, html=html)
		return subdom_funcs