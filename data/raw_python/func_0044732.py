async def log_transaction(self, **params):
		"""Writing transaction to database
		"""
		if params.get("message"):
			params = json.loads(params.get("message", "{}"))
		
		if not params:
			return {"error":400, "reason":"Missed required fields"}

		coinid = params.get("coinid")

		if not coinid in ["QTUM", "PUT"]:
			return {"error":400, "reason": "Missed or invalid coinid"}

		database = client[settings.TXS]
		source_collection = database[coinid]

		await source_collection.find_one_and_update({"txid":params.get("txid")},{"$set":{
				"blocknumber":params.get("blocknumber"),
				"blockhash":params.get("blockhash"),
				"gasLimit":params.get("gasLimit"),
				"gasPrice":params.get("gasPrice"),
			}})
		return {"success":True}