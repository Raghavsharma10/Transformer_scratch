async def get(self, cid, coinid):
		"""Receives content by content id and coin id

		Accepts:
			Query string arguments:
				- "cid" - int
				- "coinid" - str

		Returns:
			return dict with following fields:
				- "description" - str
				- "read_access" - int
				- "write_access" - int
				- "content" - str
				- "cid" - int
				- "owneraddr" - str
				- "owner" - str
				- "coinid" - str

		Verified: True
		"""
		if settings.SIGNATURE_VERIFICATION:
			super().verify()


		message = json.loads(self.get_argument("message", "{}"))

		public_key = message.get("public_key")

		# Set bridge url
		if coinid in settings.bridges.keys():
			self.account.blockchain.setendpoint(settings.bridges[coinid])
		# Get content
		content = await self.account.blockchain.getsinglecontent(cid=cid)

		if "error" in content.keys():
			self.set_status(content["error"])
			self.write(content)
			raise tornado.web.Finish 
		# Get owners account
		account = await self.account.getaccountbywallet(wallet=content["owneraddr"])
		if "error" in account.keys():
			self.set_status(account["error"])
			self.write(account)
			raise tornado.web.Finish

		# Check if it is write or read access for content
		cids = await self.account.getuserscontent(public_key=public_key)

		deals = await self.account.getdeals(buyer=public_key)


		if int(content["cid"]) in [i[0] for i in cids.get(coinid,[])]:
			content["access_type"] = "write_access"

		elif int(content["cid"]) in [i[0] for i in deals.get(coinid,[])]:
			content["access_type"] = "read_access"

		try:
			offer = await self.account.blockchain.getoffer(cid=cid, 
							buyer_address=self.account.validator[coinid](public_key))

			content["owner"] = account.get("public_key")
			content["seller_access_string"] = offer.get("seller_access_string")
			content["seller_pubkey"] = offer.get("seller_public_key")
		except:
			pass

		self.write(content)