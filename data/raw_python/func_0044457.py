async def get_frozen(self, *args, **kwargs):
		"""
		Get frozen users balance

		Accepts:
			- uid [integer] (users id)
			- types [list | string] (array with needed types or "all")

		Returns:
			{
				type [string] (blockchain type): amount
			}
		"""
		super().validate(*args, **kwargs)

		if kwargs.get("message"):
			kwargs = json.loads(kwargs.get("message"))

		# Get daya from request
		coinids = kwargs.get("coinids")
		uid = kwargs.get("uid")
		address = kwargs.get("address")
		# Check if required fields exists

		try:
			coinid = coinid.replace("TEST", "")
		except:
			pass

		try:
			uid = int(uid)
		except:
			return await self.error_400("User id must be integer. ")

		if not uid and address:
			uid = await self.get_uid_by_address(address=address, coinid=coinid)
			if isinstance(uid, dict):
				return uid

		if not all([types, uid]):
			return await self.error_400("Get frozen. Missed required fields.")
		if isinstance(types, list):
			actives = {}
			for coinid in coinids:
				database = self.client[self.collection]
				collection = database[coinid]
				# Get current balance
				balance = await collection.find_one({"uid":uid})
				if not balance:
					return await self.error_404(
						"Get frozen. Balance with uid:%s and type:%s not found" % (uid, coinid))
				# Collect actives
				actives[coinid] = int(balance["amount_frozen"])

		if isinstance(coinids, str):
			actives = {}
			for coinid in self.types:
				database = self.client[coinid]
				collection = database[self.collection]
				# Get current balance
				balance = await collection.find_one({"uid":uid})
				if not balance:
					return await self.error_404(
						"Get frozen. Balance with uid:%s and type:%s not found" % (uid, coinid))
				# Collect actives
				actives[coinid] = int(balance["amount_frozen"])
		return actives