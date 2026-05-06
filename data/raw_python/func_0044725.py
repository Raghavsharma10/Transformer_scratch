async def get_reviews(self, **params):
		"""Receives all reviews by cid
		Accepts:
		- cid
		- coinid
		"""
		if params.get("message"):
			params = json.loads(params.get("message", "{}"))
		
		if not params:
			return {"error":400, "reason":"Missed required fields"}

		cid = params.get("cid", 0)
		coinid = params.get("coinid")
		if not cid and not coinid:
			return {"error":400, "reason":"Missed cid"}

		reviews = []
		database = client[coinid]
		collection = database[settings.REVIEW]
		async for document in collection.find({"confirmed":None, "cid":int(cid)}):
			reviews.append({i:document[i] for i in document if i == "confirmed"})

		return reviews