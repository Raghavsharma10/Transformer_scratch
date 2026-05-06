async def post(self, public_key, coinid):
		"""Writes content to blockchain

		Accepts:
			Query string args:
				- "public_key" - str
				- "coin id" - str
			Request body arguments:
				- message (signed dict as json):
					- "cus" (content) - str
					- "description" - str
					- "read_access" (price for read access) - int
					- "write_access" (price for write access) - int
				- signature

		Returns:
			- dictionary with following fields:
				- "owneraddr" - str
				- "description" - str
				- "read_price" - int
				- "write_price" - int

		Verified: True
		"""
		logging.debug("[+] -- Post content debugging. ")
		#if settings.SIGNATURE_VERIFICATION:
		#	super().verify()

		# Define genesis variables
		if coinid in settings.bridges.keys():     # Define bridge url
			owneraddr = self.account.validator[coinid](public_key)    # Define owner address
			logging.debug("\n\n          Owner address")
			logging.debug(coinid)
			logging.debug(owneraddr)
			self.account.blockchain.setendpoint(settings.bridges[coinid])
		else:
			self.set_status(400)
			self.write({"error":400, "reason":"Invalid coinid"})
			raise tornado.web.Finish 


		# Check if account exists
		account = await self.account.getaccountdata(public_key=public_key)
		logging.debug("\n            Users account ")
		logging.debug(account)
		if "error" in account.keys():
			self.set_status(account["error"])
			self.write(account)
			raise tornado.web.Finish



		# Get message from request 
		try:
			data = json.loads(self.request.body)
		except:
			self.set_status(400)
			self.write({"error":400, "reason":"Unexpected data format. JSON required"})
			raise tornado.web.Finish

		if isinstance(data["message"], str):
			message = json.loads(data["message"])
		elif isinstance(data["message"], dict):
			message = data["message"]
		cus = message.get("cus")
		description = message.get("description")
		read_access = message.get("read_access")
		write_access = message.get("write_access")

		if sys.getsizeof(cus) > 1000000:
			self.set_status(403)
			self.write({"error":400, "reason":"Exceeded the content size limit."})
			raise tornado.web.Finish

		# Set fee
		fee = await billing.upload_content_fee(cus=cus, owneraddr=owneraddr, 
											description=description)
		
		if "error" in fee.keys():
			self.set_status(fee["error"])
			self.write(fee)
			raise tornado.web.Finish

		# Send request to bridge
		data = {"cus":cus, 
				"owneraddr":owneraddr, 
				"description":description, 
				"read_price":read_access,
				"write_price":write_access
				}
		response = await self.account.blockchain.makecid(**data)
		logging.debug("\n     Bridge makecid")
		logging.debug(response)
		if "error" in response.keys():
			self.set_status(400)
			self.write(response)
			raise tornado.web.Finish

		# Write cid to database
		db_content = await self.account.setuserscontent(public_key=public_key,hash=response["cus_hash"],
							coinid=coinid, txid=response["result"]["txid"],access="content")
		logging.debug("\n               Database content")
		logging.debug(db_content)


		response = {i:data[i] for i in data if i != "cus"}
		self.write(response)