async def mailed_confirm(self, **params):
		"""Sends mail to user after offer receiveing
		Accepts:
			- cid
			- buyer address
			- price
			- offer_type
			- point
			- coinid
		"""
		if not params:
			return {"error":400, "reason":"Missed required fields"}

		# Check if required fields exists
		cid = params.get("cid")
		buyer_address = params.get("buyer_address")
		price = params.get("price")
		offer_type = params.get("offer_type")
		coinid = params.get("coinid").upper()
		try:
			coinid = coinid.replace("TEST", "")
		except:
			pass
		# Check if required fileds 
		if not all([cid, buyer_address, price]):
			return {"error":400, "reason":"Missed required fields"}




		# Get content owner address
		#if coinid in settings.AVAILABLE_COIN_ID:
		#	client_bridge.endpoint = settings.bridges[coinid]
		#else:
		#	return {"error":400, "reason":"Invalid coin ID"}
		#owneraddr = await client_bridge.request(method_name="ownerbycid", cid=cid)


		# Send appropriate mail to seller if exists
		#seller = await getaccountbywallet(wallet=owneraddr)
		#logging.debug(seller)
		#if "error" in seller.keys():
		#	return seller
		#if seller.get("email"):
		#	emaildata = {
		#		"to": seller["email"],
		#		"subject": "Robin8 support",
	 	#		"optional": "You`ve got a new offer from %s" % seller["public_key"]
	 	#
		#	}
		#	await client_email.request(method_name="sendmail", **emaildata)

		# Send news for seller
		buyer = await getaccountbywallet(wallet=buyer_address) 
		if "error" in buyer.keys():
			buyer["public_key"] = None

		newsdata = {
			"event_type":"made offer",
			"cid": cid,
			"access_string":buyer["public_key"],
			"buyer_pubkey":buyer["public_key"],
			"buyer_address":buyer_address,
			#"owneraddr":owneraddr,
			"price": price,
			"offer_type": offer_type,
			"coinid":coinid
		}
		news = await self.insert_news(**newsdata)
		return {"result":"ok"}