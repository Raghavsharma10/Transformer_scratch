async def log_source(self, **params):
		""" Logging users request sources
		"""
		if params.get("message"):
			params = json.loads(params.get("message", "{}"))
		
		if not params:
			return {"error":400, "reason":"Missed required fields"}

		# Insert new source if does not exists the one

		database = client[settings.DBNAME]
		source_collection = database[settings.SOURCE]
		await source_collection.update({"public_key":params.get("public_key")}, 
						 {"$addToSet":{"source":params.get("source")}},
						 upsert=True)

		return {"result": "ok"}