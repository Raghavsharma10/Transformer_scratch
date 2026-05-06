def CSS_addRule(self, styleSheetId, ruleText, location):
		"""
		Function path: CSS.addRule
			Domain: CSS
			Method name: addRule
		
			Parameters:
				Required arguments:
					'styleSheetId' (type: StyleSheetId) -> The css style sheet identifier where a new rule should be inserted.
					'ruleText' (type: string) -> The text of a new rule.
					'location' (type: SourceRange) -> Text position of a new rule in the target style sheet.
			Returns:
				'rule' (type: CSSRule) -> The newly created rule.
		
			Description: Inserts a new rule with the given <code>ruleText</code> in a stylesheet with given <code>styleSheetId</code>, at the position specified by <code>location</code>.
		"""
		assert isinstance(ruleText, (str,)
		    ), "Argument 'ruleText' must be of type '['str']'. Received type: '%s'" % type(
		    ruleText)
		subdom_funcs = self.synchronous_command('CSS.addRule', styleSheetId=
		    styleSheetId, ruleText=ruleText, location=location)
		return subdom_funcs