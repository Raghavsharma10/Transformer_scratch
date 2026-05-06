async def put(self, cid):
		"""Update price of current content

		Accepts:
			Query string args:
				- "cid" - int
			Request body params: 
				- "access_type" - str
				- "price" - int
				- "coinid" - str

		Returns:
			dict with following fields:
				- "confirmed": None
				- "txid" - str
				- "description" - str
				- "content" - str
				- "read_access" - int
				- "write_access" - int
				- "cid" - int
				- "txid" - str
				- "seller_pubkey" - str
				- "seller_access_string": None or str

		Verified: True

		""" 
		if settings.SIGNATURE_VERIFICATION:
			super().verify()

		try:
			body = json.loads(self.request.body)
		except:
			self.set_status(400)
			self.write({"error":400, "reason":"Unexpected data format. JSON required"})
			raise tornado.web.Finish

		# Get data from signed message	
		public_key = body.get("public_key", None)
		if isinstance(body["message"], str):
			message = json.loads(body["message"])
		elif isinstance(body["message"], dict):
			message = body["message"]
		price = message.get("price")
		access_type = message.get("access_type")
		coinid = message.get("coinid")

		# Check if required fields exists
		if not any([price, access_type, coinid]):
			self.set_status(400)
			self.write({"error":400, "reason":"Missed price and access type for content"})

		# Set bridges url
		if coinid in settings.bridges.keys():
			self.account.blockchain.setendpoint(settings.bridges[coinid])
		else:
			self.set_status(400)
			self.write({"error":400, "reason":"Invalid coin ID"})
			raise tornado.web.Finish

		# Get public key hex or checksum format
		check = self.account.validator[coinid](public_key)

		# Get content owner address
		owneraddr = await self.account.blockchain.ownerbycid(cid=cid)
		if isinstance(owneraddr, dict):
			if "error" in owneraddr.keys():
				self.set_status(404)
				self.write({"error":404, "reason":"Owner not found."})
				raise tornado.web.Finish

		# Check if current content belongs to current user
		if owneraddr != check:
			self.set_status(403)
			self.write({"error":403, "reason":"Owner does not match."})
			raise tornado.web.Finish

		response = {"cid":cid, "coinid":coinid}

		# Make setprice request to the bridge
		if access_type == "write_price":
			result = await self.account.blockchain.setwriteprice(cid=cid, write_price=price)
			response["write_access"] = result["price"]

		elif access_type == "read_price":
			result = await self.account.blockchain.setreadprice(cid=cid, read_price=price)
			response["read_access"] = result["price"]


		# Fee
		fee = await billing.set_price_fee(cid=cid, price=price, owneraddr=owneraddr)
		if "error" in fee.keys():
			self.set_status(fee["error"])
			self.write(fee)
			raise tornado.web.Finish

		self.write(response)