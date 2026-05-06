async def put(self, cid):
		"""Update description for content

		Accepts:
			Query string args:
				- "cid" - int
			Request body parameters:
				- message (signed dict):
					- "description" - str
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
		descr = message.get("description")
		coinid = message.get("coinid")

		if not coinid in settings.bridges.keys():
			self.set_status(400)
			self.write({"error":400, "reason":"Unknown coin id"})
			raise tornado.web.Finish

		# Check if all required data exists
		if not all([public_key, descr, coinid]):
			self.set_status(400)
			self.write({"error":400, "reason":"Missed required fields"})
			raise tornado.web.Finish
		
		owneraddr = self.account.validator[coinid](public_key)

		# Get content owner
		response = await self.account.blockchain.ownerbycid(cid=cid)
		if isinstance(response, dict):
			if "error" in response.keys():
				error_code = response["error"]
				self.set_status(error_code)
				self.write({"error":error_code, "reason":response["error"]})
				raise tornado.web.Finish

		# Check if current content belongs to current user
		if response != owneraddr:
			self.set_status(403)
			self.write({"error":403, "reason":"Owner does not match."})
			raise tornado.web.Finish

		# Set fee
		fee = await billing.update_description_fee(owneraddr=owneraddr,cid=cid, 
													description=descr)

		# Set bridge url
		if coinid in settings.bridges.keys():
			self.account.blockchain.setendpoint(settings.bridges[coinid])
		else:
			self.set_status(400)
			self.write({"error":400, "reason":"Invalid coinid"})
			raise tornado.web.Finish 
		
		# Set description for content. Make request to the bridge
		request = await self.account.blockchain.setdescrforcid(cid=cid, descr=descr, 
																owneraddr=owneraddr)
		if "error" in request.keys():
			self.set_status(request["error"])
			self.write(request)
			raise tornado.web.Finish

		self.write({"cid":cid, "description":descr, 
					"coinid":coinid, "owneraddr": owneraddr})