async def update_contents(self, **params):
		"""Updates users content row
		Accepts:
		- txid
		- cid
		- description
		- write_price
		- read_price
		- confirmed
		- coinid
		"""
		if params.get("message"):
			params = json.loads(params.get("message", "{}"))

		if not params:
			return {"error":400, "reason":"Missed required fields"}

		txid = params.get("txid")

		coinid = params.get("coinid").upper()

		try:
			coinid = coinid.replace("TEST", "")
		except:
			pass

		database = client[coinid]
		content_collection = database[settings.CONTENT]

		content = await content_collection.find_one({"txid":txid})

		if not content:
			return {"error":404, 
					"reason":"Update content. Content with txid %s not found" % txid}

		if content.get("hash"):
			self.account.blockchain.setendpoint(settings.bridges[coinid])
			cid = await self.account.blockchain.getcid(hash=content["hash"])

			await content_collection.find_one_and_update({"txid":txid}, {"$set":{"cid":int(cid)}})
			await content_collection.find_one_and_update({"txid":txid}, {"$set":{"hash":None}})


		updated = await content_collection.find_one({"txid":txid})

		return {i:updated[i] for i in updated if i != "_id"}