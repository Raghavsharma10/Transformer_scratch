def trace(self, context, obj):
		"""Enumerate the children of the given object, as would be visible and utilized by dispatch."""
		
		root = obj
		
		if isroutine(obj):
			yield Crumb(self, root, endpoint=True, handler=obj, options=opts(obj))
			return
		
		for name, attr in getmembers(obj if isclass(obj) else obj.__class__):
			if name == '__getattr__':
				sig = signature(attr)
				path = '{' + list(sig.parameters.keys())[1] + '}'
				reta = sig.return_annotation
				
				if reta is not sig.empty:
					if callable(reta) and not isclass(reta):
						yield Crumb(self, root, path, endpoint=True, handler=reta, options=opts(reta))
					else:
						yield Crumb(self, root, path, handler=reta)
				
				else:
					yield Crumb(self, root, path, handler=attr)
				
				del sig, path, reta
				continue
			
			elif name == '__call__':
				yield Crumb(self, root, None, endpoint=True, handler=obj)
				continue
			
			if self.protect and name[0] == '_':
				continue
			
			yield Crumb(self, root, name,
					endpoint=callable(attr) and not isclass(attr), handler=attr, options=opts(attr))