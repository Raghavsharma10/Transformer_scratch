async def get(self, public_key):
		"""Retrieves all users input and output offers
		Accepts:
		- public key
		"""
		# Sign-verifying functional
		if settings.SIGNATURE_VERIFICATION:
			super().verify()

		logging.debug("\n\n --  Input offers debugging")

		message = json.loads(self.get_argument("message"))
		cid = message.get("cid")
		coinid = message.get("coinid")
		if not cid:
			self.set_status(400)
			self.write({"error":400, "reason":"Missed required fields."})
			raise tornado.web.Finish


		account = await self.account.getaccountdata(public_key=public_key)
		if "error" in account.keys():
			self.set_status(account["error"])
			self.write(account)
			raise tornado.web.Finish

		if coinid in settings.bridges.keys():
			self.account.blockchain.setendpoint(settings.bridges[coinid])
		offers = await self.account.blockchain.getcidoffers(cid=cid)
		logging.debug("\n\n -- Offers")
		logging.debug(offers)

		if isinstance(offers, dict):
			self.set_status(offers["error"])
			self.write(offers)
			raise tornado.web.Finish

		for offer in offers:
			offer["type"] = self.account.ident_offer[offer["type"]]

		storage_offers = await self.account.getoffers(cid=cid, coinid=coinid)
		logging.debug("\n\n -- Storage offers. ")
		logging.debug(storage_offers)

		self.write(json.dumps(offers + storage_offers))