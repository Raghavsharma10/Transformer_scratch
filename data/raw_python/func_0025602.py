def transform(self, context, handler, result):
		"""Transform the value returned by the controller endpoint.
		
		This extension transforms returned values if the endpoint has a return type annotation.
		"""
		handler = handler.__func__ if hasattr(handler, '__func__') else handler
		annotation = getattr(handler, '__annotations__', {}).get('return', None)
		
		if annotation:
			return (annotation, result)
		
		return result