async def insert_offer(self, **params):
		"""Inserts new offer to database (related to buyers account)
		Accepts:
			- cid
			- buyer address
			- price
			- access type
			- transaction id
			- owner public key
			- owner address
			- coin ID
		"""
		if params.get("message"):
			params = json.loads(params.get("message", "{}"))
		
		if not params:
			return {"error":400, "reason":"Missed required fields"}

		# Check if required fields exists
		cid = int(params.get("cid", 0))
		txid = params.get("txid")
		coinid = params.get("coinid")
		public_key = params.get("public_key")

		database = client[coinid]
		offer_collection = database[settings.OFFER]
		await offer_collection.insert_one({"cid":cid, "txid":txid, 
											"confirmed":None, "coinid":coinid, 
											"public_key":public_key})

		return {"result":"ok"}