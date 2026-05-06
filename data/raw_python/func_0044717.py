async def get_offers(self, **params):
		"""Receives all users input (by cid) or output offers 
		Accepts:
		- public key
		- cid (optional)
		- coinid (optional)
		"""
		if params.get("message"):
			params = json.loads(params.get("message", "{}"))
		
		if not params:
			return {"error":400, "reason":"Missed required fields"}

		cid = params.get("cid")
		public_key = params.get("public_key")
		coinid = params.get("coinid")


		# Get all input offers by cid
		if cid and coinid:
			cid = int(cid)
			database = client[coinid]
			offer_collection = database[settings.OFFER]
			content_collection = database[settings.CONTENT]

			offers = [{i:document[i] for i in document if i == "confirmed"} 
						async for document in offer_collection.find({"cid":cid, "confirmed":None})]

		# Get all output users offers
		elif not cid:
			database = client[coinid]
			offer_collection = database[settings.OFFER]
			offers = [{i:document[i] for i in document if i == "confirmed"} 
						async for document in offer_collection.find({"public_key":public_key, 
																	"confirmed":None})]

		# Return list with offers
		return offers