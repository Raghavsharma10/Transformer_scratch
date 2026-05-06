async def remove_offer(self, **params):
		"""Receives offfer after have deal
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
		buyer_address = params.get("buyer_address")
		coinid = params.get("coinid")

		try:
			coinid = coinid.replace("TEST", "")
		except:
			pass
	
		# Check if required fileds 
		if not all([cid, buyer_address]):
			return {"error":400, "reason":"Missed required fields"}
		
		# Try to find offer with account id and cid	
		offer = await self.get_offer(cid=cid, buyer_address=buyer_address, coinid=coinid)
		if "error" in offer.keys():
			return offer

		# Remove offer
		database = client[coinid]
		offer_collection = database[settings.OFFER]
		await offer_collection.delete_one(
							{"account_id":offer["account_id"],
							"cid":cid})
		return {"result": "ok"}