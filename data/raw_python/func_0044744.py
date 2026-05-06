async def post(self, public_key):
		"""Writes contents review
		"""
		if settings.SIGNATURE_VERIFICATION:
			super().verify()

		try:
			body = json.loads(self.request.body)
		except:
			self.set_status(400)
			self.write({"error":400, "reason":"Unexpected data format. JSON required"})
			raise tornado.web.Finish
			
		if isinstance(body["message"], str):
			message = json.loads(body["message"])
		elif isinstance(body["message"], dict):
			message = body["message"]

		cid = message.get("cid")
		review = message.get("review")
		rating = message.get("rating")
		coinid = message.get("coinid")

		if not all([cid, rating, review]):
			self.set_status(400)
			self.write({"error":400, "reason":"Missed required fields"})

		if coinid in settings.bridges.keys():
			self.account.blockchain.setendpoint(settings.bridges[coinid])
		else:
			self.set_status(400)
			self.write({"error":400, "reason":"Invalid coinid"})
			raise tornado.web.Finish 

		buyer_address = self.account.validator[coinid](public_key)

		review = await self.account.blockchain.addreview(cid=int(cid),buyer_address=buyer_address,
														stars=int(rating), review=review)
		await self.account.setreview(cid=cid, txid=review["result"]["txid"], coinid=coinid)

		self.write({"cid":cid, "review":review, "rating":rating})