async def read(self, *_id):
		"""Read data from database table.
		Accepts ids of entries.
		Returns list of results if success
			or string with error code and explanation.

		read(*id) => [(result), (result)] (if success)
		read(*id) => [] (if missed)
		read() => {"error":400, "reason":"Missed required fields"}
		"""
		if not _id:
			return {"error":400, 
					"reason":"Missed required fields"}

		result = []
		for i in _id:
			document = await self.collection.find_one({"id":i})
			try:
				result.append({i:document[i] for i in document
												if i != "_id"})
			except:
				continue
		return result