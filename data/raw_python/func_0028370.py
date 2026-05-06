def pop(self, key, default = None):
		"""Delete the item"""
		node = self.get(key, None)
		
		if node == None:
			value = default
		else:
			value = node
		try:
			del self[key]
		except:
			return value
		return value