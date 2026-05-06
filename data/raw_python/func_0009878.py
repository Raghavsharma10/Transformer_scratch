def Audits_getEncodedResponse(self, requestId, encoding, **kwargs):
		"""
		Function path: Audits.getEncodedResponse
			Domain: Audits
			Method name: getEncodedResponse
		
			Parameters:
				Required arguments:
					'requestId' (type: Network.RequestId) -> Identifier of the network request to get content for.
					'encoding' (type: string) -> The encoding to use.
				Optional arguments:
					'quality' (type: number) -> The quality of the encoding (0-1). (defaults to 1)
					'sizeOnly' (type: boolean) -> Whether to only return the size information (defaults to false).
			Returns:
				'body' (type: string) -> The encoded body as a base64 string. Omitted if sizeOnly is true.
				'originalSize' (type: integer) -> Size before re-encoding.
				'encodedSize' (type: integer) -> Size after re-encoding.
		
			Description: Returns the response body and size if it were re-encoded with the specified settings. Only applies to images.
		"""
		assert isinstance(encoding, (str,)
		    ), "Argument 'encoding' must be of type '['str']'. Received type: '%s'" % type(
		    encoding)
		if 'quality' in kwargs:
			assert isinstance(kwargs['quality'], (float, int)
			    ), "Optional argument 'quality' must be of type '['float', 'int']'. Received type: '%s'" % type(
			    kwargs['quality'])
		if 'sizeOnly' in kwargs:
			assert isinstance(kwargs['sizeOnly'], (bool,)
			    ), "Optional argument 'sizeOnly' must be of type '['bool']'. Received type: '%s'" % type(
			    kwargs['sizeOnly'])
		expected = ['quality', 'sizeOnly']
		passed_keys = list(kwargs.keys())
		assert all([(key in expected) for key in passed_keys]
		    ), "Allowed kwargs are ['quality', 'sizeOnly']. Passed kwargs: %s" % passed_keys
		subdom_funcs = self.synchronous_command('Audits.getEncodedResponse',
		    requestId=requestId, encoding=encoding, **kwargs)
		return subdom_funcs