def CSS_setMediaText(self, styleSheetId, range, text):
		"""
		Function path: CSS.setMediaText
			Domain: CSS
			Method name: setMediaText
		
			Parameters:
				Required arguments:
					'styleSheetId' (type: StyleSheetId) -> No description
					'range' (type: SourceRange) -> No description
					'text' (type: string) -> No description
			Returns:
				'media' (type: CSSMedia) -> The resulting CSS media rule after modification.
		
			Description: Modifies the rule selector.
		"""
		assert isinstance(text, (str,)
		    ), "Argument 'text' must be of type '['str']'. Received type: '%s'" % type(
		    text)
		subdom_funcs = self.synchronous_command('CSS.setMediaText', styleSheetId=
		    styleSheetId, range=range, text=text)
		return subdom_funcs