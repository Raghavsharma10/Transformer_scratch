def Browser_setWindowBounds(self, windowId, bounds):
		"""
		Function path: Browser.setWindowBounds
			Domain: Browser
			Method name: setWindowBounds
		
			Parameters:
				Required arguments:
					'windowId' (type: WindowID) -> Browser window id.
					'bounds' (type: Bounds) -> New window bounds. The 'minimized', 'maximized' and 'fullscreen' states cannot be combined with 'left', 'top', 'width' or 'height'. Leaves unspecified fields unchanged.
			No return value.
		
			Description: Set position and/or size of the browser window.
		"""
		subdom_funcs = self.synchronous_command('Browser.setWindowBounds',
		    windowId=windowId, bounds=bounds)
		return subdom_funcs