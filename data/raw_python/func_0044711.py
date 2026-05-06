async def getaccountbywallet(**params):
	"""Receives account by wallet
	Accepts:
	- public key hex or checksum format
	"""
	if params.get("message"):
		params = json.loads(params.get("message"))

	for coinid in coin_ids:

		database = client[coinid]
		wallet_collection = database[settings.WALLET]
		wallet = await wallet_collection.find_one({"wallet":params["wallet"]})
		if not wallet:
			continue
		else:
			database = client[settings.DBNAME]
			accounts_collection = database[settings.ACCOUNTS]
			account = await accounts_collection.find_one({"id":wallet["account_id"]})
			if not account:
				return {"error":404, "reason":"Account was not found"}
			return {i:account[i] for i in account if i != "_id"}
	else:
		return {"error":404, "reason":"Account was not found"}