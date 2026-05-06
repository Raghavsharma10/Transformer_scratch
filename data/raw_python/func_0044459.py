async def get_wallets(self, *args, **kwargs):
		"""
		Get users wallets by uid

		Accepts:
			- uid [integer] (users id)

		Returns a list:
			- [
					{
						"address": [string],
						"uid": [integer],
						"amount_active": [integer],
						"amount_frozen": [integer]
					},
				]
		"""
		logging.debug("\n [+] -- Get wallets debugging.")
		if kwargs.get("message"):
			kwargs = json.loads(kwargs.get("message"))
		logging.debug(kwargs)
		uid = kwargs.get("uid",0)
		address = kwargs.get("address")
		coinid = kwargs.get("coinid")

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

		wallets = [i async for i in self.collect_wallets(uid)]

		return {"wallets":wallets}