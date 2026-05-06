def CSS_setKeyframeKey(self, styleSheetId, range, keyText):
		"""
		Function path: CSS.setKeyframeKey
			Domain: CSS
			Method name: setKeyframeKey
		
			Parameters:
				Required arguments:
					'styleSheetId' (type: StyleSheetId) -> No description
					'range' (type: SourceRange) -> No description
					'keyText' (type: string) -> No description
			Returns:
				'keyText' (type: Value) -> The resulting key text after modification.
		
			Description: Modifies the keyframe rule key text.
		"""
		assert isinstance(keyText, (str,)
		    ), "Argument 'keyText' must be of type '['str']'. Received type: '%s'" % type(
		    keyText)
		subdom_funcs = self.synchronous_command('CSS.setKeyframeKey',
		    styleSheetId=styleSheetId, range=range, keyText=keyText)
		return subdom_funcs