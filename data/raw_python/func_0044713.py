async def find_recent_news(self, **params):
		"""Looking up recent news for account.
		Accepts:
			- public_key
		Returns:
			- list with dicts or empty
		"""
		# Check if params is not empty
		if params.get("message"):
			params = json.loads(params.get("message", "{}"))
		if not params:
			return {"error":400, "reason":"Missed required fields"}

		# Check if required parameter does exist
		public_key = params.get("public_key", None)
		if not public_key:
			return {"error":400, "reason":"Missed required fields"}

		# Check if current public_key does exist in database
		account = await self.collection.find_one({"public_key": public_key})
		if not account:
			return {"error":404, "reason":"Current user not found"}

		
		# Connect to news collection
		news_db = client[settings.DBNAME]
		news_collection = news_db[settings.NEWS]
		
		news = [{i:new[i] for i in new if i != "_id"} 
					async for new in news_collection.find(
						{"account_id":account["id"]}).sort([("$natural", -1)])]

		
		# Set news amount to zero.
		accounts_collection = news_db[settings.ACCOUNTS]
		await accounts_collection.find_one_and_update(
						{"public_key": params["public_key"]},
						{"$set": {"news_count": 0}})
		return news