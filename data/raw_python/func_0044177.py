async def insert(self, **kwargs):
		"""
		Accepts request object, retrieves data from the one`s body
		and creates new account. 
		"""
		
		if kwargs:
			# Create autoincrement for account
			pk = await self.autoincrement()
			kwargs.update({"id": pk})

			# Create account with received data and autoincrement
			await self.collection.insert_one(kwargs)

			row = await self.collection.find_one({"id": pk})

		else:
			row = None

		if row:
			return {i:row[i] for i in row if i != "_id"}
		else:
			return {"error":500, 
					"reason":"Not created"}