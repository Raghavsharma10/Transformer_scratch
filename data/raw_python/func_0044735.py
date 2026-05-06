async def get(self):
		"""
		Accepts:
			without parameters

		Returns:
			list of dictionaries with following fields:
				- "description" - str
				- "read_access" - int
				- "write_access" - int
				- "cid" - int
				- "owneraddr" - str
				- "coinid" - str

		Verified: False
		"""
		logging.debug("\n\n All Content debugging --- ")

		page = self.get_query_argument("page", 1)

		contents = []

		coinids = list(settings.bridges.keys())

		logging.debug("\n\n Coinids ")
		logging.debug(coinids)

		for coinid in coinids:
			logging.debug("\n [=] -- coinid")
			logging.debug(coinid)

			if coinids.index(coinid) == len(coinids) - 1:
				paginator = Paginator(coinid=coinid, page=page, 
					limit=(settings.LIMIT//len(coinids))+(settings.LIMIT%len(coinids)))
			else:
				paginator = Paginator(coinid=coinid, page=page, 
									limit=settings.LIMIT // len(coinids))


			self.account.blockchain.setendpoint(settings.bridges[coinid])

			
			content = await self.account.blockchain.getallcontent(
										range_=paginator.get_range())

			if isinstance(content, dict):
				if "error" in content.keys():
					continue
			contents.extend(content)
		
		response = {
			"profiles":json.dumps(contents),
		}
		try:
			response.update(paginator.get_pages())
		except:
			pass

		self.write(json.dumps(response))