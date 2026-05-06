async def update_description(self, **params):
		"""Set description to unconfirmed status
		after updating by user.
		Accepts:
		- cid
		- description
		- transaction id
		- coinid
		"""
		if params.get("message"):
			params = json.loads(params.get("message", "{}"))
		
		if not params:
			return {"error":400, "reason":"Missed required fields"}

		# Check if required fields exists
		cid = params.get("cid")
		description = params.get("description")
		txid = params.get("txid")
		coinid = params.get("coinid")

		try:
			coinid = coinid.replace("TEST", "")
		except:
			pass

		# Check if required fileds 
		if not all([cid, description, txid, coinid]):
			return {"error":400, "reason":"Missed required fields"}

		# Try to find offer with account id and cid
		database = client[coinid]
		collection = database[settings.CONTENT]
		content = await collection.find_one({"cid":int(cid)})
		if not content:
			return {"error":404, 
					"reason":"Content with cid %s not found" % cid }

		# Update offer
		await collection.find_one_and_update(
							{"cid":int(cid)}, {"$set":{"description":description}})
		await collection.find_one_and_update(
							{"cid":int(cid)}, {"$set":{"confirmed":None}})
		await collection.find_one_and_update(
							{"cid":int(cid)}, {"$set":{"txid":txid}})

		# Get updated offer
		updated = await collection.find_one({"cid":int(cid)})

		return {i:updated[i] for i in updated if i != "_id"}