async def get_offer(self, **params):
		"""Receives offer data if exists
		Accepts:
			- cid
			- buyer address
			- coin ID
		"""
		if params.get("message"):
			params = json.loads(params.get("message", "{}"))
		
		if not params:
			return {"error":400, "reason":"Missed required fields"}

		# Check if required fields exists
		cid = int(params.get("cid", 0))
		coinid = params.get("coinid")
		buyer_address = params.get("buyer_address")
		
		# Check if required fileds 
		if not all([cid, buyer_address, coinid]):
			return {"error":400, "reason":"Missed required fields"}
		
		# Get buyer address row from database
		database = client[coinid]
		wallet_collection = database[settings.WALLET]
		wallet = await wallet_collection.find_one({"wallet":buyer_address})
		if not wallet:
			return {"error":404, "reason":"Buyer address not found"}
		
		# Try to find offer with account id and cid
		offer_collection = database[settings.OFFER]
		offer = await offer_collection.find_one(
							{"account_id":int(wallet["account_id"]),
							"cid":int(cid)})

		# If current offer exists avoid creating a new one
		if not offer:
			return {"error":404, "reason": "Current offer not found"}
		else:
			offer["coinid"] = coinid
			return {i:offer[i] for i in offer if i != "_id"}