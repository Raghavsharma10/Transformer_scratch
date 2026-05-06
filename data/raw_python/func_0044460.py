async def confirmbalance(self, *args, **kwargs):
		""" Confirm balance after trading

		Accepts:
		    - message (signed dictionary):
		        - "txid" - str
		        - "coinid" - str
		        - "amount" - int

		Returns:
		        - "address" - str
		        - "coinid" - str
		        - "amount" - int
		        - "uid" - int
		        - "unconfirmed" - int (0 by default)
		        - "deposit" - int (0 by default)

		Verified: True

		"""
		# Get data from request
		if kwargs.get("message"):
			kwargs = json.loads(kwargs.get("message", "{}"))

		txid = kwargs.get("txid")
		coinid = kwargs.get("coinid")
		buyer_address = kwargs.get("buyer_address")
		cid = kwargs.get("cid")
		address = kwargs.get("buyer_address")

		try:
			coinid = coinid.replace("TEST", "")
		except:
			pass

		# Check if required fields exists
		if not all([coinid, cid, buyer_address, txid]):
		    return {"error":400, "reason": "Confirm balance. Missed required fields"}

		if not coinid in settings.bridges.keys():
			return await self.error_400("Confirm balance. Invalid coinid: %s" % coinid)

		# Get offers price	
		self.account.blockchain.setendpoint(settings.bridges[coinid])
		offer = await self.account.blockchain.getoffer(cid=cid, 
											buyer_address=buyer_address)
		# Get offers price for updating balance
		amount = int(offer["price"])

		coinid = "PUT"
		# Get sellers account
		history_database = self.client[settings.HISTORY]
		history_collection = history_database[coinid]
		history = await history_collection.find_one({"txid":txid})

		try:
			account = await self.account.getaccountdata(public_key=history["public_key"])
		except:
			return await self.error_404("Confirm balance. Not found current deal.")

		# Connect to balance database
		database = self.client[self.collection]
		balance_collection = database[coinid]

		# Try to update balance if exists
		balance = await balance_collection.find_one({"uid":account["id"]})
		# Decrement unconfirmed
		submitted = int(balance["amount_frozen"]) - int(amount)
		if submitted < 0:
			return await self.error_400("Not enough frozen amount.")

		decremented = await balance_collection.find_one_and_update(
		                        {"uid":account["id"]}, 
		                        {"$set":{"amount_frozen": str(submitted)}})

		difference = int(balance["amount_active"]) + int(amount)
		updated = await balance_collection.find_one_and_update(
		                        {"uid":account["id"]}, 
		                        {"$set":{"amount_active":str(difference)}})
		if not updated:
		    return {"error":404, 
		            "reason":"Confirm balance. Not found current transaction id"}

		# Delete transaction id field
		await history_collection.find_one_and_update({"txid":txid}, 
												{"$unset":{"txid":1}})


		if int(account["level"]) == 2:
		    await self.account.updatelevel(**{"id":account["id"], "level":3})

		return {i:updated[i] for i in updated if i != "_id" and i != "txid"}