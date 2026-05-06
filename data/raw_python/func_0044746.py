async def post(self):
		"""Creates new account

		Accepts:
			- message (signed dict):
				- "device_id" - str
				- "email" - str
				- "phone" - str
			- "public_key" - str
			- "signature" - str

		Returns:
			dictionary with following fields:
				- "device_id" - str
				- "phone" - str
				- "public_key" - str
				- "count" - int  ( wallets amount )
				- "level" - int (2 by default)
				- "news_count" - int (0 by default)
				- "email" - str
				- "href" - str
				- "wallets" - list

		Verified: True

		"""
		logging.debug("\n\n[+] -- Account debugging. ")
		# Include signature verification mechanism
		if settings.SIGNATURE_VERIFICATION:
			super().verify()

		# Save data at storage database
		try:
			data = json.loads(self.request.body)
		except:
			self.set_status(400)
			self.write({"error":400, "reason":"Unexpected data format. JSON required"})
			raise tornado.web.Finish
		message = data["message"]

		# Create account
		new_account = await self.account.createaccount(**data)
		logging.debug("\n\n [+] -- New account debugging.")
		logging.debug(new_account["id"])
		if "error" in new_account.keys():
			# Raise error if the one does exist
			self.set_status(new_account["error"])
			self.write(new_account)
			raise tornado.web.Finish

		# Receive balance from balance host
		wallets = await self.account.balance.get_wallets(uid=new_account["id"])
		if isinstance(wallets, dict):
			if "error" in wallets.keys():
				self.set_status(wallets["error"])
				self.write(wallets)
				raise tornado.web.Finish

		#Prepare response 
		new_account.update({"href": settings.ENDPOINTS["ams"]+"/"+ new_account["public_key"],
							"wallets": json.dumps(wallets["wallets"])})
		# Send mail to user
		if new_account.get("email"):
			email_data = {
				"to": new_account["email"],
	        	"subject": "Robin8 Support",
	         	"optional": "Your account was created on %s" % settings.domain + new_account["href"]
	        }
			await self.account.mailer.sendmail(**email_data)
		# Response
		self.write(new_account)