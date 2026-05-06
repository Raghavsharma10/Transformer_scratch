def CSS_setStyleSheetText(self, styleSheetId, text):
		"""
		Function path: CSS.setStyleSheetText
			Domain: CSS
			Method name: setStyleSheetText
		
			Parameters:
				Required arguments:
					'styleSheetId' (type: StyleSheetId) -> No description
					'text' (type: string) -> No description
			Returns:
				'sourceMapURL' (type: string) -> URL of source map associated with script (if any).
		
			Description: Sets the new stylesheet text.
		"""
		assert isinstance(text, (str,)
		    ), "Argument 'text' must be of type '['str']'. Received type: '%s'" % type(
		    text)
		subdom_funcs = self.synchronous_command('CSS.setStyleSheetText',
		    styleSheetId=styleSheetId, text=text)
		return subdom_funcs