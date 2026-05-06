async def update_offer(self, **params):
		"""Updates offer after transaction confirmation
		Accepts:
			- transaction id
			- coinid
			- confirmed (boolean flag)
		"""
		logging.debug("\n\n -- Update offer. ")
		if params.get("message"):
			params = json.loads(params.get("message", "{}"))
		
		if not params:
			return {"error":400, "reason":"Missed required fields"}

		# Check if required fields exists
		txid = params.get("txid")
		coinid = params.get("coinid").upper()

		try:
			coinid = coinid.replace("TEST", "")
		except:
			pass

		# Try to find offer with account id and cid
		database = client[coinid]
		offer_db = database[settings.OFFER]
		offer = await offer_db.find_one({"txid":txid})
		logging.debug("\n\n -- Try to get offer. ")
		logging.debug(offer)
		if not offer:
			return {"error":404, 
					"reason":"Offer with txid %s not found" % txid }

		# Update offer
		await offer_db.find_one_and_update(
							{"txid":txid}, {"$set":{"confirmed":1}})

		# Get updated offer
		updated = await offer_db.find_one({"txid":txid})

		return {i:updated[i] for i in updated if i != "_id"}