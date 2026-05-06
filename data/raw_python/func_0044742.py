async def post(self, public_key):
		"""Accepting offer by buyer

		Function accepts:
			- cid
			- buyer access string
			- buyer public key
			- seller public key
		"""
		logging.debug("[+] -- Deal debugging. ")
		if settings.SIGNATURE_VERIFICATION:
			super().verify()

		# Check if message contains required data
		try:
			body = json.loads(self.request.body)
		except:
			self.set_status(400)
			self.write({"error":400, "reason":"Unexpected data format. JSON required"})
			raise tornado.web.Finish
		logging.debug("\n         Body")
		logging.debug(body)
			
		if isinstance(body["message"], str):
			message = json.loads(body["message"])
		elif isinstance(body["message"], dict):
			message = body["message"]

		cid = message.get("cid")
		buyer_pubkey = message.get("buyer_pubkey")
		buyer_access_string = message.get("buyer_access_string")

		seller_access_string = message.get("seller_access_string")
		access_type = message.get("access_type")

		coinid = message.get("coinid")
		# check passes data
		if not all([buyer_access_string, cid, buyer_pubkey, coinid]):
			self.set_status(400)
			self.write({"error":400, "reason":"Missed required fields"})
			raise tornado.web.Finish

		if coinid in settings.bridges.keys():
			self.account.blockchain.setendpoint(settings.bridges[coinid])

		else:
			self.set_status(400)
			self.write({"error":400, "reason":"Invalid coin ID"})
			raise tornado.web.Finish 

		# Sellcontent
		buyer_address = self.account.validator[coinid](buyer_pubkey)

		# Check if accounts exists
		seller_account = await self.account.getaccountdata(public_key=public_key)
		logging.debug("\n          Seller account")
		logging.debug(seller_account)

		try:
			error_code = seller_account["error"]
		except:
			pass
		else:
			self.set_status(error_code)
			self.write(seller_account)
			raise tornado.web.Finish

		buyer_account = await self.account.getaccountdata(public_key=buyer_pubkey)
		logging.debug("\n           Buyer account")
		logging.debug(buyer_account)
		try:
			error_code = buyer_account["error"]
		except:
			pass
		else:
			self.set_status(error_code)
			self.write(buyer_account)
			raise tornado.web.Finish

		# Check if content belongs to current account
		owneraddr = await self.account.blockchain.ownerbycid(cid=cid)
		
		
		if owneraddr != self.account.validator[coinid](public_key):
			self.set_status(403)
			self.write({"error":403, "reason":"Forbidden. Profile owner does not match."})
			raise tornado.web.Finish

		#Get buyers balance
		balances = await self.account.balance.get_wallets(coinid=coinid, 
														uid=buyer_account["id"])
		if isinstance(balances, dict):
			if "error" in balances.keys():
				self.set_status(balances["error"])
				self.write(balances)
				raise tornado.web.Finish 


		# Get difference with balance and price
		get_price = await self.account.blockchain.getoffer(buyer_address=buyer_address,
															cid=cid)
		price = get_price["price"]
		logging.debug("\n          Price")
		logging.debug(price)

		for w in balances["wallets"]:
			if "PUT" in w.values() or "PUTTEST" in w.values():
				balance = w

		difference = int(balance["amount_frozen"]) - int(price)

		if difference >= 0:

			if access_type == "write_access":
				logging.debug("\n          Write access")
				# Fee
				fee = await billing.change_owner_fee(cid=cid, new_owner=buyer_pubkey)
				if "error" in fee.keys():
					self.set_status(fee["error"])
					self.write(fee)
					raise tornado.web.Finish
			
				# Change content owner
				chownerdata = {
					"cid":cid,
					"new_owner": buyer_address,
					"access_string": buyer_access_string,
					"seller_public_key": public_key
				}

				response = await self.account.blockchain.changeowner(**chownerdata)
				logging.debug("\n         Bridge change owner")
				logging.debug(response)

				new_owner = await self.account.changeowner(cid=cid, 
										public_key=buyer_account["public_key"], 
										coinid=coinid)
				logging.debug("\n Database new owner")
				logging.debug(new_owner)

			elif access_type == "read_access":
				logging.debug("\n          Read access")

				# Fee
				fee = await billing.sell_content_fee(cid=cid, new_owner=buyer_pubkey)

				if "error" in fee.keys():
					self.set_status(fee["error"])
					self.write(fee)
					raise tornado.web.Finish

				selldata = {
					"cid":cid,
					"buyer_address":buyer_address,
					"access_string":buyer_access_string,
					"seller_public_key":public_key
				}

				response = await self.account.blockchain.sellcontent(**selldata)
				logging.debug("\n        Bridge sell content")

				# Write cid to database
				check = await self.account.setuserscontent(public_key=buyer_account["public_key"], 
											hash=None,coinid=coinid, cid=cid,
											txid=response["result"]["txid"],
											access="deal")
				logging.debug(check)

			
			# Increment and decrement balances of seller and buyer
			coinid = "PUT"
			unfreeze = await self.account.balance.unfreeze(uid=buyer_account["id"],
													amount=price, coinid=coinid)
			logging.debug("\n          Unfreeze buyer")
			logging.debug(unfreeze)

			sub_active = await self.account.balance.sub_active(uid=buyer_account["id"], 
												coinid=coinid, amount=price)
			logging.debug("\n            Sub active")
			logging.debug(sub_active)
			add_frozen = await self.account.balance.add_frozen(uid=seller_account["id"], 
												amount=price, coinid=coinid)
			logging.debug("\n               Add frozen")
			logging.debug(add_frozen)
			# Write entry with txid to database
			new_deal = await self.account.balance.registerdeal(uid=seller_account["id"],
													public_key=seller_account["public_key"], 
													txid=response["result"]["txid"],
													coinid=coinid, cid=cid)

			del response["result"]
			del response["contract_owner_hex"]
			self.write(response)
		else:
			# If Insufficient funds
			self.set_status(402)
			self.write({"error":402, "reason":"Insufficient funds of buyer"})
			raise tornado.web.Finish