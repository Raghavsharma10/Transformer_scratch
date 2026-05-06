async def get_active(self, *args, **kwargs):
		"""
		Get active users balance

		Accepts:
			- uid [integer] (users id)
			- types [list | string] (array with needed types or "all")

		Returns:
			{
				type [string] (blockchain type): amount
			}
		"""

		# Get daya from request
		coinids = kwargs.get("coinids")
		uid = kwargs.get("uid",0)
		address = kwargs.get("address")

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

		# Check if required fields exists
		if not all([coinids, uid]):
			return await self.error_400("Get active. Missed required fields.")
		if isinstance(coinids, list):
			actives = {}
			for coinid in coinids:
				database = self.client[self.collection]
				collection = database[coinid]
				# Get current balance
				balance = await collection.find_one({"uid":uid})
				if not balance:
					return await self.error_404(
						"Get active. Balance with uid:%s and type:%s not found" % (uid, coinid))
				# Collect actives
				actives[coinid] = int(balance["amount_active"])

		if isinstance(coinids, str):
			actives = {}
			for coinid in self.coinids:
				database = self.client[coinid]
				collection = database[self.collection]
				# Get current balance
				balance = await collection.find_one({"uid":uid})
				if not balance:
					return await self.error_404(
						"Get active. Balance with uid:%s and type:%s not found" % (uid, coinid))
				# Collect actives
				actives[coinid] = int(balance["amount_active"])
		return actives