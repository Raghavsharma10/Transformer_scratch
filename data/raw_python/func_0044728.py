async def write_deal(self, **params):
		"""Writes deal to database
		Accepts:
		- cid
		- access_type
		- buyer public key
		- seller public key
		- price
		- coinid
		"""
		if params.get("message"):
			params = json.loads(params.get("message", "{}"))
		
		if not params:
			return {"error":400, "reason":"Missed required fields"}

		cid = int(params.get("cid", 0))
		access_type = params.get("access_type")
		buyer = params.get("buyer")
		seller = params.get("seller")
		price = params.get("price")
		coinid = params.get("coinid")

		try:
			coinid = coinid.replace("TEST", "")
		except:
			pass

		if not all([cid, access_type, buyer, seller, price]):
			return {"error":400, "reason":"Missed required fields"}

		database = client[coinid]
		collection = database[settings.DEAL]
		await collection.insert_one({
				"cid":cid,
				"access_type": access_type,
				"buyer":buyer,
				"seller":seller,
				"price":price,
				"coinid":coinid
			})
		result = await collection.find_one({"cid":cid, "buyer":buyer})

		return {i:result[i] for i in result if i != "_id"}