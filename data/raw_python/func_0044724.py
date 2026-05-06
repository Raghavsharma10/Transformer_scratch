async def set_access_string(self, **params):
		"""Writes content access string to database 
		"""
		if params.get("message"):
			params = json.loads(params.get("message", "{}"))

		cid = int(params.get("cid", "0"))
		seller_access_string = params.get("seller_access_string")
		seller_pubkey = params.get("seller_pubkey")
		coinid = params.get("coinid")

		try:
			coinid = coinid.replace("TEST", "")
		except:
			pass

		database = client[coinid]
		collection = database[settings.CONTENT]
		content = await collection.find_one({"cid":cid})

		if not content:
			return {"error":404, "reason":"Content not found"}

		if not all([cid, seller_access_string, seller_pubkey]):
			return {"error":400, "reason":"Missed required fields"}

		await collection.find_one_and_update({"cid":cid},
						{"$set":{"seller_access_string":seller_access_string}})

		await collection.find_one_and_update({"cid":cid},
						{"$set":{"seller_pubkey":seller_pubkey}})

		content = await collection.find_one({"cid":cid})
		return {i:content[i] for i in content if i != "_id"}