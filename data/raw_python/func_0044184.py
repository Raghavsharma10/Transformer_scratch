def verify(self):
		"""Abstract method.
		Signature verifying logic.

		"""
		logging.debug("\n\n")
		logging.debug("[+] -- Verify debugging")
		logging.debug("\n\n")

		if self.request.body:
			logging.debug("\n Request body")
			logging.debug(self.request.body)
			data = json.loads(self.request.body)
			message = json.dumps(data.get("message")).replace(" ", "")
			logging.debug("\n")
			logging.debug(message)

		elif self.request.arguments:
			logging.debug("\n Arguments")
			logging.debug(self.request.arguments)
			data = {i:self.get_argument(i) for i in self.request.arguments}
			message = data.get("message", "{}")
			logging.debug(message)

		try:
			# Check if required fields exist
			assert "public_key" in data.keys(), "Missed public key in parameters"
			assert "message" in data.keys(), "Missed message in parameters"
			assert "signature" in data.keys(),"Missed signature in parameters"
			public_key = data["public_key"]
			signature = data["signature"]
			timestamp = data.get("timestamp", None)
			
			# Check if
			#assert ManagementSystemHandler.get_time_stamp() == timestamp, "Timestamps does not match. Try again."

		except Exception as e:
			self.set_status(403)
			self.write({"error":403, "reason": "Missing signature " + str(e)})
			raise tornado.web.Finish

		else:
			# Check if message and signature exist
			# If not - return 403 error code
			if not all([message, public_key, signature]):
				raise tornado.web.HTTPError(403)
		# If exist - call verifying static method
		try:
			logging.debug("\n[] Try block. Verifying")
			logging.debug(message)
			logging.debug(signature)
			logging.debug(public_key)
			flag = Qtum.verify_message(message, signature, public_key)
		except Exception as e:
			# If public key is not valid or it`s missing - return 404 error
			#self.set_status(403)
			#self.write({"error":403, 
			#			"reason":"Forbidden. Invalid signature." + str(e)})
			#raise tornado.web.Finish
			logging.debug("\n Exception")
			logging.debug(str(e))
			pass