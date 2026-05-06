async def set_review(self, **params):
		"""Writes review for content
		Accepts:
		- cid
		- review
		- public_key
		- rating
		- txid
		- coinid
		"""
		if params.get("message"):
			params = json.loads(params.get("message", "{}"))
		
		if not params:
			return {"error":400, "reason":"Missed required fields"}

		cid = int(params.get("cid", 0))
		txid = params.get("txid")
		coinid = params.get("coinid")

		try:
			coinid = coinid.replace("TEST", "")
		except:
			pass
		
		# Get content
		database = client[coinid]
		content_collection = database[settings.CONTENT]
		content = await content_collection.find_one({"cid":cid})
		if not content:
			return {"error":404, "reason":"Not found current content"}

		database = client[coinid]
		review_collection = database[settings.REVIEW]		
		await review_collection.insert_one({"cid":cid, "confirmed":None, 
											"txid":txid, "coinid":coinid})
		return {"result":"ok"}