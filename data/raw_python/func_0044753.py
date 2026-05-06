async def upload_content_fee(*args, **kwargs):
	"""Estimating uploading content
	"""
	cus = kwargs.get("cus")
	owneraddr = kwargs.get("owneraddr")
	description = kwargs.get("description")
	coinid = kwargs.get("coinid", "PUT")

	#Check if required fields exists
	if not all([cus, owneraddr, description]):
		return {"error":400, "reason":"Missed required fields"}

	# Get upload content fee
	content_fee = billing.estimate_upload_fee(len(cus))
	descr_fee = billing.estimate_set_descr_fee(len(description))
	# Get users account
	user = await client_storage.request(method_name="getaccountbywallet", wallet=owneraddr)
	if "error" in user.keys():
		return user
	# Get users balance
	balances = await client_balance.request(method_name="get_wallets", uid=user["id"])
	logging.debug("[+] -- Balances debugging.")
	logging.debug(balances)
	if isinstance(balances, dict):
		if "error" in balances.keys():
			return balances
	# Decrement users balance
	common_price = int(content_fee) + int(descr_fee)

	for w in balances["wallets"]:
		if coinid in w.values():
			balance = w
	diff = int(balance["amount_active"]) - common_price
	if diff < 0:
		return {"error":403, "reason": "Balance is not enough."}

	decbalance = await client_balance.request(method_name="sub_active", uid=user["id"],
	                        						coinid=coinid, amount=common_price)
	if "error" in decbalance.keys():
		return decbalance
	else:
		return {"result":"ok"}