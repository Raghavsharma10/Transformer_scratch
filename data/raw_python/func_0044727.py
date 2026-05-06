async def update_review(self, **params):
		"""Update review after transaction confirmation
		Accepts:
			- txid
			- coinid
		"""
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
		collection = database[settings.REVIEW]
		review = await collection.find_one({"txid":txid})
		if not review:
			return {"error":404, 
					"reason":"Review with txid %s not found" % txid }
		
		# Update review
		await collection.find_one_and_update(
							{"txid":txid}, {"$set":{"confirmed":1}})
		
		# Get updated offer
		updated = await collection.find_one({"txid":txid})

		return {i:updated[i] for i in updated if i != "_id"}