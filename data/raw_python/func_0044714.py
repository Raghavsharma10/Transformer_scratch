async def insert_news(self, **params):
		"""Inserts news for account
		Accepts:
			- event_type
			- cid
			- access_string (of buyer)
			- buyer_pubkey
			- buyer address
			- owner address
			- price
			- offer type
			- coin ID
		Returns:
			- dict with result
		"""
		logging.debug("\n\n [+] -- Setting news debugging.  ")
		if params.get("message"):
			params = json.loads(params.get("message", "{}"))
		
		if not params:
			return {"error":400, "reason":"Missed required fields"}
		logging.debug("   ***      Params")
		event_type = params.get("event_type")
		cid = params.get("cid")

		access_string = params.get("access_string")

		buyer_pubkey = params.get("buyer_pubkey")

		buyer_address = params.get("buyer_address")

		owneraddr = params.get("owneraddr")

		price = params.get("price")

		offer_type = int(params.get("offer_type", -1))

		coinid = params.get("coinid").upper()

		try:
			coinid = coinid.replace("TEST", "")
		except:
			pass

		logging.debug("\n **   Coinid")
		logging.debug(coinid)

		# Get address of content owner and check if it exists
		if coinid in settings.bridges.keys():
			self.account.blockchain.setendpoint(settings.bridges[coinid])
		else:
			return {"error":400, "reason": "Invalid coin ID"}

		owneraddr = await self.account.blockchain.ownerbycid(cid=cid)

		# Get sellers account
		seller = await getaccountbywallet(wallet=owneraddr)
		if "error" in seller.keys():
			return seller

		# Connect to news table 
		news_collection = self.database[settings.NEWS]

		# Get sellers price
		self.account.blockchain.setendpoint(settings.bridges[coinid])
		if offer_type == 1:
			seller_price = await self.account.blockchain.getwriteprice(cid=cid)
		elif offer_type == 0:
			seller_price = await self.account.blockchain.getreadprice(cid=cid)

		
		row = {"offer_type": self.account.ident_offer[offer_type], 
				"buyer_address":buyer_address,
				"cid":cid,
				"access_string":access_string,
				"buyer_pubkey": buyer_pubkey,
				"seller_price": seller_price,
				"buyer_price": price,
				"account_id": seller["id"],
				"event_type": event_type,
				"coinid":coinid}

		logging.debug("\n **  Inserting row")
		logging.debug(row)
		
		# Update counter inside accounts table
		database = client[settings.DBNAME]
		collection = database[settings.ACCOUNTS]
		await collection.find_one_and_update(
						{"id": int(seller["id"])},
						{"$inc": {"news_count": 1}})
		await collection.find_one({"id":int(seller["id"])})
		
		# Insert data to news table
		await news_collection.insert_one(row)

		logging.debug("\n ** Fresh news")
		fresh = await collection.find_one({"buyer_address":buyer_address,
												"cid":cid})
		logging.debug(fresh)

		return {"result":"ok"}