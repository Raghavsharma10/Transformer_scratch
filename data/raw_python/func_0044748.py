async def get(self, public_key):
		"""Receives public key, looking up document at storage,
				sends document id to the balance server
		"""
		if settings.SIGNATURE_VERIFICATION:
			super().verify()

		response = await self.account.getnews(public_key=public_key)
		# If we`ve got a empty or list with news 
		if isinstance(response, list):
			self.write(json.dumps(response))
			raise tornado.web.Finish
		# If we`ve got a message with error
		elif isinstance(response, dict):
			try:
				error_code = response["error"]
			except:
				del response["account_id"]

				self.write(response)
			else:
				self.set_status(error_code)
				self.write(response)
				raise tornado.web.Finish