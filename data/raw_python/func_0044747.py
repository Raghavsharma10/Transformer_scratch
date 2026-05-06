async def get(self, public_key):
		""" Receive account data

		Accepts:
			Query string:
				- "public_key" - str
			Query string params:
				- message ( signed dictionary ):
					- "timestamp" - str
	
		Returns:
				- "device_id" - str
				- "phone" - str
				- "public_key" - str
				- "count" - int  ( wallets amount )
				- "level" - int (2 by default)
				- "news_count" - int (0 by default)
				- "email" - str
				- "wallets" - list
		
		Verified: True

		"""
		# Signature verification
		if settings.SIGNATURE_VERIFICATION:
			super().verify()

		# Get users request source
		compiler = re.compile(r"\((.*?)\)")
		match = compiler.search(self.request.headers.get("User-Agent"))
		try:
			source = match.group(1)
		except:
			source = None
		# Write source to database
		await self.account.logsource(public_key=public_key, source=source)

		# Get account
		logging.debug("\n\n [+] -- Get account data.")
		response = await self.account.getaccountdata(public_key=public_key)
		logging.debug("\n")
		logging.debug(response)
		logging.debug("\n")
		if "error" in response.keys():
			self.set_status(response["error"])
			self.write(response)
			raise tornado.web.Finish

		# Receive balances from balance host
		wallets = await self.account.balance.get_wallets(uid=response["id"])
		if isinstance(wallets, dict):
			if "error" in wallets.keys():
				self.set_status(wallets["error"])
				self.write(wallets)
				raise tornado.web.Finish
		
		# Filter wallets
		response.update({"wallets":json.dumps(
					[i for i in wallets["wallets"] 
					if i.get("coinid") not in ["BTC", "LTC", "ETH"]])})
		# Return account data
		self.write(response)