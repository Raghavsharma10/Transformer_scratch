async def delete(self, _id=None):
		"""Delete entry from database table.
		Accepts id.
		delete(id) => 1 (if exists)
		delete(id) => {"error":404, "reason":"Not found"} (if does not exist)
		delete() => {"error":400, "reason":"Missed required fields"}
		"""
		if not _id:
			return {"error":400, 
					"reason":"Missed required fields"}

		document = await self.collection.find_one({"id": _id})

		if not document:
			return {"error":404, 
					"reason":"Not found"}

		deleted_count = await self.collection.delete_one(
							{"id": _id}).deleted_count

		return deleted_count