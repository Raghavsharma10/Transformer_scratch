async def get(self, public_key):
		"""Retrieves all users contents
		Accepts:
		- public key
		"""
		# Sign-verifying functional
		if settings.SIGNATURE_VERIFICATION:
			super().verify()

		page = self.get_query_argument("page", 1)

		cids = await self.account.getuserscontent(public_key=public_key)

		logging.debug("\n\n Users cids")
		logging.debug(cids)
		
		if isinstance(cids, dict):
			if "error" in cids.keys():
				self.set_status(cids["error"])
				self.write(cids)
				raise tornado.web.Finish

		container = []

		for coinid in cids:

			logging.debug("\n [] -- coinid")
			logging.debug(coinid)

			#if list(cids.keys()).index(coinid) == len(cids) - 1:
			#	paginator = Paginator(coinid=coinid, page=page, 
			#		limit=(settings.LIMIT//len(cids))+(settings.LIMIT%len(cids)), cids=cids)
			#else:
			#paginator = Paginator(coinid=coinid, page=page, 
			#						limit=settings.LIMIT // len(cids), cids=cids)

			if coinid in settings.bridges.keys():
				logging.debug(" -- Coinid in ")
				logging.debug(settings.bridges.keys())
				self.account.blockchain.setendpoint(settings.bridges[coinid])

				contents = await self.account.blockchain.getuserscontent(
												cids=json.dumps(cids[coinid]))
				logging.debug("\n\n -- Contents")
				logging.debug(contents)
				if isinstance(contents, dict):
					if "error" in contents.keys():
						continue
				container.extend(contents)

				logging.debug("\n\n -- Container 1")


				logging.debug("\n\n -- Container 2")
				logging.debug(container)

		response = {
			"profiles":json.dumps(container),
			}
		try:
			response.update(paginator.get_pages())
		except:
			pass
	
		self.write(json.dumps(response))