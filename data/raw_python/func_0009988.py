def get_unpacked_response_body(self, requestId, mimetype="application/unknown"):
		'''
		Return a unpacked, decoded resposne body from Network_getResponseBody()
		'''
		content = self.Network_getResponseBody(requestId)

		assert 'result' in content
		result = content['result']

		assert 'base64Encoded' in result
		assert 'body' in result

		if result['base64Encoded']:
			content = base64.b64decode(result['body'])
		else:
			content = result['body']

		self.log.info("Navigate complete. Received %s byte response with type %s.", len(content), mimetype)

		return {'binary' : result['base64Encoded'],  'mimetype' : mimetype, 'content' : content}