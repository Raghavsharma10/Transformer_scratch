async def post(self, public_key=None):
		"""Creates new offer

		Accepts:
			- buyer public key
			- cid
			- buyer access string
		Returns:
			- offer parameters as dictionary
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
		read_price = message.get("price")
		coinid = message.get("coinid")
		buyer_access_string = message.get("buyer_access_string")
	
		if not all([buyer_access_string, coinid, str(cid).isdigit()]):
			self.set_status(400)
			self.write({"error":400, "reason":"Missed required fields"})
			raise tornado.web.Finish

		# Set bridge url
		if coinid in settings.bridges.keys():
			self.account.blockchain.setendpoint(settings.bridges[coinid])
		else:
			self.set_status(400)
			self.write({"error":400, "reason":"Invalid coinid"})
			raise tornado.web.Finish 

		# Get cid price from bridge
		if not read_price:
			read_price = await self.account.blockchain.getreadprice(cid=cid)

		buyer_address = self.account.validator[coinid](public_key)
		owneraddr = await self.account.blockchain.ownerbycid(cid=cid)

		# Check if public key exists
		account = await self.account.getaccountdata(public_key=public_key)
		if "error" in account.keys():
			# If account does not exist
			self.set_status(account["error"])
			self.write(account)
			raise tornado.web.Finish

		#Get sellers balance
		balances = await self.account.balance.get_wallets(coinid=coinid, uid=account["id"])

		# Check if current content does not belong to current user
		if owneraddr == buyer_address:
			self.set_status(400)
			self.write({"error":400, 
						"reason":"Content belongs to current user"})
			raise tornado.web.Finish

		# Get difference with balance and price
		for w in balances["wallets"]:
			if "PUT" in w.values() or "PUTTEST" in w.values():
				balance = w

		difference = int(balance["amount_active"]) - int(read_price)
		if difference < 0:
			# If Insufficient funds
			self.set_status(402)
			self.write({"error":402, "reason":"Balance is not enough"})
			raise tornado.web.Finish

		# Send request to bridge
		offer_data = {
			"cid":cid,
			"read_price":read_price,
			"offer_type":0,
			"buyer_address": buyer_address,
			"buyer_access_string":buyer_access_string
		}
		response = await self.account.blockchain.makeoffer(**offer_data)
		try:
			response["error"]
		except:
			pass
		else:
			self.set_status(response["error"])
			self.write(response)
			raise tornado.web.Finish

		await self.account.insertoffer(cid=cid, txid=response["result"]["txid"], 
											coinid=coinid, public_key=public_key)
		# Send e-mail to seller
		seller = await self.account.getaccountbywallet(wallet=owneraddr)
		if "error" in seller.keys():
			self.set_status(seller["error"])
			self.write(seller)
			raise tornado.web.Finish

		if seller.get("email"):
			emaildata = {
				"to": seller["email"],
				"subject": "Robin8 support",
     			"optional": "You`ve got an offer for content %s." % cid
			}
			await self.account.mailer.sendmail(**emaildata)


		# Freeze price at balance
		coinid = "PUT"
		await self.account.balance.freeze(uid=account["id"],coinid=coinid, 
																amount=read_price)

		# Set fee
		fee = await billing.set_make_offer_fee(buyer_address=buyer_address)
		if "error" in fee.keys():
			self.set_status(fee["error"])
			self.write(fee)
			raise tornado.web.Finish

		response["offer_type"] = "read_access"
		del response["result"]		

		self.write(response)