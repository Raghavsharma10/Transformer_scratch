async def get(self, public_key):
		"""Retrieves all users input and output offers
		Accepts:
		- public key
		"""
		# Sign-verifying functional
		#super().verify()
		# Get coinid
		account = await self.account.getaccountdata(public_key=public_key)
		if "error" in account.keys():
			self.set_status(account["error"])
			self.write(account)
			raise tornado.web.Finish



		offers_collection = []
		for coinid in settings.AVAILABLE_COIN_ID:

			try:
				self.account.blockchain.setendpoint(settings.bridges[coinid])
			except:
				continue

			try:
				offers = await self.account.blockchain.getbuyeroffers( 
								buyer_address=self.account.validator[coinid](public_key))
				for offer in offers:
					offer["type"] = self.account.ident_offer[offer["type"]]
	
				storage_offers = await self.account.getoffers(coinid=coinid, 
																public_key=public_key)

			except:
				continue

			offers_collection.extend(offers + storage_offers)


		self.write(json.dumps(offers_collection))