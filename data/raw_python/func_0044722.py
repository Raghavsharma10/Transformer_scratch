async def set_contents(self, **params):
		"""Writes users content to database
		Accepts:
		- public key (required)
		- content (required)
		- description
		- price
		- address
		"""
		if params.get("message"):
			params = json.loads(params.get("message", "{}"))
		
		if not params:
			return {"error":400, "reason":"Missed required fields"}

		txid = params.get("txid")
		public_key = params.get("public_key")
		_hash = params.get("hash")
		coinid = params.get("coinid")
		access = params.get("access")
		cid = params.get("cid")

		# Try to get account
		account = await self.collection.find_one({"public_key":public_key})
		# Return error if does not exist the one
		if not account:
			return {"error":404, "reason":"Account was not found"}

		database = client[coinid]
		content_collection = database[access]
		await content_collection.insert_one({
								"owner": public_key,
								"cid":cid,
								"txid": txid, 
								"hash": _hash
						})
		success = await content_collection.find_one({"txid":txid})
		if not success:
			return {"error":500, "reason":"Error while writing content to database"}

		else:
			return {"result":"ok"}