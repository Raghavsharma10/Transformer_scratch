async def get(self, cid, coinid):
		"""Receives all contents reviews
		"""
		if settings.SIGNATURE_VERIFICATION:
			super().verify()

		if coinid in settings.bridges.keys():
			self.account.blockchain.setendpoint(settings.bridges[coinid])


		reviews = await self.account.blockchain.getreviews(cid=cid)
		if isinstance(reviews, dict):
			if "error" in reviews:
				self.set_status(500)
				self.write(reviews)
				raise tornado.web.Finish

		for review in reviews:
			review["confirmed"] = 1

		storage_reviews = await self.account.getreviews(coinid=coinid, cid=cid)

		if isinstance(reviews, dict):
			if "error" in reviews.keys():
				self.set_status(reviews["error"])
				self.write(reviews)
				raise tornado.web.Finish
		
		self.write(json.dumps(reviews + storage_reviews))