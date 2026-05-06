async def post(self):
		"""
		Funds from account to given address.
		1. Verify signature
		2. Freeze senders amount.
		3. Request to withdraw server.
		4. Call balances sub_frozen method.

		Accepts:
			- message [dict]:
				- coinid [string]
				- amount [integer]
				- address [string]
				- timestamp [float]
				- recvWindow [float]
			- public_key
			- signature

		Returns:
			- message [dict]:
				- coinid [string]
				- amount [integer]
				- address [string]
				- timestamp [float]
				- recvWindow [float]
			- public_key
			- signature
			- txid [string]
		"""
		# Sign-verifying functional
		if settings.SIGNATURE_VERIFICATION:
			super().verify()

		logging.debug("\n\n[+] -- Withdraw debugging")
		# Get data from requests body
		data = json.loads(self.request.body)
		public_key = data.get("public_key")
		signature = data.get("signature")

		if isinstance(data.get("message"), str):
			message = json.loads(data["message"])
		elif isinstance(data.get("message"), dict):
			message = data["message"]

		# Get data from signed message
		coinid = message.get("coinid")
		amount = message.get("amount")
		address = message.get("address")
		timestamp = message.get("timestamp")
		recvWindow = message.get("recvWindow")
		# 
		if not all([coinid, amount, address, public_key, 
					signature, timestamp, recvWindow]):
			data.update({"error":400, "reason":"Missed required fields. "})
			self.set_status(400)
			self.write(data)
			raise tornado.web.Finish
		logging.debug(data)

		# Get account
		account = await self.account.getaccountdata(public_key=public_key)
		if "error" in account.keys():
			data.update(account)
			self.set_status(404)
			self.write(data)
			raise tornado.web.Finish
		logging.debug("\n                Senders account")
		logging.debug(account)


		# Request to balance and call freeze method
		fee = await self.account.withdraw_fee(coinid)

		freeze = await self.account.balance.freeze(uid=account["id"], coinid=coinid,
													amount=amount + fee)
		logging.debug("\n           Frozen balance")
		logging.debug(freeze)
		if "error" in freeze.keys():
			data.update(freeze)
			self.set_status(freeze["error"])
			self.write(data)
			raise tornado.web.Finish

		# Request to withdraw server
		txid = await self.account.withdraw(amount=amount, coinid=coinid, 
							address=address)
		logging.debug("\n      Withdraw server response")
		logging.debug(txid)

		# Check if txid exists
		if "error" in txid.keys():
			await self.account.balance.unfreeze(uid=account["id"], coinid=coinid,
														amount=amount + fee)
			data.update(txid)
			self.set_status(500)
			self.write(data)
			raise tornado.web.Finish

		# Add balance to recepient
		#add_active = await self.account.balance.add_active(address=address, coinid=coinid,
		#													amount=amount)
		#if "error" in add_active.keys():
		#	await self.account.balance.unfreeze(uid=account["id"], coinid=coinid,
		#												amount=amount + fee)
		#	data.update(add_active)
		#	self.set_status(add_active["error"])
		#	self.write(data)
		#	raise tornado.web.Finish

		# Submit amount from frozen balance
		sub_frozen = await self.account.balance.sub_frozen(uid=account["id"], 
													coinid=coinid, amount=amount + fee)
		if "error" in sub_frozen.keys():			
		
			data.update(sub_frozen)
			self.set_status(sub_frozen["error"])
			self.write(data)
			raise tornado.web.Finish
		logging.debug("\n               Sub frozen")
		logging.debug(sub_frozen)

		await self.account.save_transaction(txid=txid.get("txid"), coinid=coinid,
													amount=amount, address=address)

		# Return txid
		data.update(txid)
		self.write(data)