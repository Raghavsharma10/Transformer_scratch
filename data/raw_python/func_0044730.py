async def get_deals(self, **params):
		"""Receives all users deals
		Accepts:
		- buyer public key
		"""
		if params.get("message"):
			params = json.loads(params.get("message", "{}"))
		
		if not params:
			return {"error":400, "reason":"Missed required fields"}

		buyer = params.get("buyer")

		if not buyer:
			return {"error":400, "reason":"Missed public key"}

		deals = {i:[] for i in list(settings.bridges.keys())}

		for coinid in list(settings.bridges.keys()):
			database = client[coinid]
			collection = database[settings.DEAL]
			async for document in collection.find({"owner":buyer}):
				deals[coinid].append((document["cid"],document.get("txid")))
		return deals