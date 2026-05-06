async def put(self, public_key):
		"""Reject offer and unfreeze balance
		
		Accepts:
			- cid
			- buyer public key
			- buyer address
		"""
		if settings.SIGNATURE_VERIFICATION:
			super().verify()

		# Check if message contains required data
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
		cid = int(message["offer_id"].get("cid", 0))
		buyer_address = message["offer_id"].get("buyer_address")
		coinid = message.get("coinid")

		if not all([cid, buyer_address, coinid]):
			self.set_status(400)
			self.write({"error":400, "reason": "Missed required fields."})
			raise tornado.web.Finish

		if coinid in settings.bridges.keys():
			self.account.blockchain.setendpoint(settings.bridges[coinid])
		else:
			self.set_status(400)
			self.write({"error":400, "reason":"Invalid coin ID"})
			raise tornado.web.Finish

		check = self.account.validator[coinid](public_key)

		account = await self.account.getaccountdata(public_key=public_key)
		if "error" in account.keys():
			error_code = account["error"]
			self.set_status(error_code)
			self.write(account)
			raise tornado.web.Finish
	
		# Check if one of sellers or buyers rejects offer
		owneraddr = await self.account.blockchain.ownerbycid(cid=cid)
		hex_ = check
		if buyer_address != hex_ and owneraddr != hex_:
			# Avoid rejecting offer
			self.set_status(403)
			self.write({"error": 403, "reason":"Forbidden. Offer does not belong to user."})
		
		# Reject offer
		response = await self.account.blockchain.rejectoffer(coinid=coinid, cid=cid, 
															buyer_address=buyer_address)
		if "error" in response.keys():
			self.set_status(response["error"])
			self.write(response)
			raise tornado.web.Finish

		# Get buyer for email sending
		buyer = await self.account.getaccountbywallet(wallet=buyer_address)
		if "error" in buyer.keys():
			self.set_status(buyer["error"])
			self.write(buyer)
			raise tornado.web.Finish
	
		if buyer.get("email"):
			emaildata = {
				"to": buyer.get("email"),
				"subject": "Robin8 support",
     			"optional": "Your offer with cid %s was rejected." % cid
			}
			await self.account.mailer.sendmail(**emaildata)
		
		# Undeposit balance
		price = await self.account.blockchain.getwriteprice(cid=cid)
		coinid = "PUT"
		await self.account.balance.unfreeze(uid=buyer["id"],coinid=coinid, 
													amount=price)

		del response["result"]
		self.write(response)