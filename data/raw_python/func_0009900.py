def DOMStorage_setDOMStorageItem(self, storageId, key, value):
		"""
		Function path: DOMStorage.setDOMStorageItem
			Domain: DOMStorage
			Method name: setDOMStorageItem
		
			Parameters:
				Required arguments:
					'storageId' (type: StorageId) -> No description
					'key' (type: string) -> No description
					'value' (type: string) -> No description
			No return value.
		
		"""
		assert isinstance(key, (str,)
		    ), "Argument 'key' must be of type '['str']'. Received type: '%s'" % type(
		    key)
		assert isinstance(value, (str,)
		    ), "Argument 'value' must be of type '['str']'. Received type: '%s'" % type(
		    value)
		subdom_funcs = self.synchronous_command('DOMStorage.setDOMStorageItem',
		    storageId=storageId, key=key, value=value)
		return subdom_funcs