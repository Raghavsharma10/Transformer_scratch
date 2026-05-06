async def find(self, **kwargs):
		"""Find all entries with given search key.
		Accepts named parameter key and arbitrary values.
		Returns list of entry id`s.

		find(**kwargs) => document (if exist)
		find(**kwargs) => {"error":404,"reason":"Not found"} (if does not exist)
		find() => {"error":400, "reason":"Missed required fields"}
		"""
		if not isinstance(kwargs, dict) and len(kwargs) != 1:
			return {"error":400, 
					"reason":"Bad request"}
		document = await self.collection.find_one(kwargs)
		if document:
			return document
		else:
			return {"error":404, "reason":"Not found"}