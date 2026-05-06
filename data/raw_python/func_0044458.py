async def collect_wallets(self, uid):
		"""
		Asynchronous generator
		"""
		logging.debug(self.types)
		logging.debug(uid)
		for coinid in self.types:
			logging.debug(coinid)
			await asyncio.sleep(0.5)
			# Connect to appropriate database
			database = self.client[self.collection]
			logging.debug(database)
			collection = database[coinid]
			logging.debug(collection)
			# Get wallets
			wallet = await collection.find_one({"uid":int(uid)})
			logging.debug(wallet)

			wallet["amount_active"] = int(wallet["amount_active"])
			wallet["amount_frozen"] = int(wallet["amount_frozen"])
			del wallet["_id"]
			yield wallet