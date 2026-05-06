async def get_contents(self, **params):
		"""Retrieves all users content
		Accepts:
		-public key
		"""
		logging.debug("[+] -- Get contents")
		if params.get("message"):
			params = json.loads(params.get("message", "{}"))

		if not params or not params.get("public_key"):
			return {"error":400, "reason":"Missed required fields"}

		# Try to get account
		account = await self.collection.find_one({"public_key":params["public_key"]})
		# Return error if does not exist the one
		if not account:
			return {"error":404, "reason":"Get contents. Not found account"}

		contents = {i:[] for i in settings.AVAILABLE_COIN_ID}
		for coinid in settings.AVAILABLE_COIN_ID:
			logging.debug(coinid)
			database = client[coinid]
			content_collection = database[settings.CONTENT]
			async for document in content_collection.find({"owner":account["public_key"]}):
				contents[coinid].append((document["cid"], document["txid"]))

		return contents