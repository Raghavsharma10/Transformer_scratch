async def freeze(self, *args, **kwargs):
		"""
		Freeze users balance

		Accepts:
			- uid [integer] (users id from main server)
			- coinid [string] (blockchain type in uppercase)
			- amount [integer] (amount for freezing)

		Returns:
			- uid [integer] (users id from main server)
			- coinid [string] (blockchain type in uppercase)
			- amount_active [integer] (activae users amount)
			- amount_frozen [integer] (frozen users amount)
		"""

		# Get data from request
		uid = kwargs.get("uid", 0)
		coinid = kwargs.get("coinid")
		amount = kwargs.get("amount")
		address = kwargs.get("address")

		try:
			coinid = coinid.replace("TEST", "")
		except:
			pass

		try:
			uid = int(uid)
		except:
			return await self.error_400("User id must be integer. ")

		try:
			amount = int(amount)
		except:
			return await self.error_400("Amount must be integer. ")

		try:
			assert amount > 0
		except:
			return await self.error_400("Amount must be positive integer. ")


		# Check if required fields exists
		if not uid and address:
			uid = await self.get_uid_by_address(address=address, coinid=coinid)
			if isinstance(uid, dict):
				return uid

		# Connect to appropriate database
		database = self.client[self.collection]
		collection = database[coinid]

		# Check if balance exists
		balance = await collection.find_one({"uid":uid})
		if not balance:
			return await self.error_404(
				"Freeze. Balance with uid:%s and type:%s not found." % (uid, coinid))

		# Check if amount is enough
		difference = int(balance["amount_active"]) - int(amount)
		if difference < 0:
			return await self.error_403("Freeze. Insufficient amount in the account")
		# Decrement active amount and increment frozen amount
		amount_frozen = int(balance["amount_frozen"]) + int(amount) 
		await collection.find_one_and_update({"uid":uid},
						{"$set":{"amount_active":str(difference), 
									"amount_frozen":str(amount_frozen)}})
		
		# Return updated balance with excluded mongo _id field
		result = await collection.find_one({"uid":uid})
		result["amount_frozen"] = int(result["amount_frozen"])
		result["amount_active"] = int(result["amount_active"])
		del result["_id"]

		return result