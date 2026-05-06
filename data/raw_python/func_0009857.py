def Overlay_setShowPaintRects(self, result):
		"""
		Function path: Overlay.setShowPaintRects
			Domain: Overlay
			Method name: setShowPaintRects
		
			Parameters:
				Required arguments:
					'result' (type: boolean) -> True for showing paint rectangles
			No return value.
		
			Description: Requests that backend shows paint rectangles
		"""
		assert isinstance(result, (bool,)
		    ), "Argument 'result' must be of type '['bool']'. Received type: '%s'" % type(
		    result)
		subdom_funcs = self.synchronous_command('Overlay.setShowPaintRects',
		    result=result)
		return subdom_funcs