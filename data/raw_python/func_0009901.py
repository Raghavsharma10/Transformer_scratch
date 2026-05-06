def DOMStorage_removeDOMStorageItem(self, storageId, key):
		"""
		Function path: DOMStorage.removeDOMStorageItem
			Domain: DOMStorage
			Method name: removeDOMStorageItem
		
			Parameters:
				Required arguments:
					'storageId' (type: StorageId) -> No description
					'key' (type: string) -> No description
			No return value.
		
		"""
		assert isinstance(key, (str,)
		    ), "Argument 'key' must be of type '['str']'. Received type: '%s'" % type(
		    key)
		subdom_funcs = self.synchronous_command('DOMStorage.removeDOMStorageItem',
		    storageId=storageId, key=key)
		return subdom_funcs