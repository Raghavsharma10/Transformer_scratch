def CSS_setRuleSelector(self, styleSheetId, range, selector):
		"""
		Function path: CSS.setRuleSelector
			Domain: CSS
			Method name: setRuleSelector
		
			Parameters:
				Required arguments:
					'styleSheetId' (type: StyleSheetId) -> No description
					'range' (type: SourceRange) -> No description
					'selector' (type: string) -> No description
			Returns:
				'selectorList' (type: SelectorList) -> The resulting selector list after modification.
		
			Description: Modifies the rule selector.
		"""
		assert isinstance(selector, (str,)
		    ), "Argument 'selector' must be of type '['str']'. Received type: '%s'" % type(
		    selector)
		subdom_funcs = self.synchronous_command('CSS.setRuleSelector',
		    styleSheetId=styleSheetId, range=range, selector=selector)
		return subdom_funcs