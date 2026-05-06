def get_user_by_email(self, email):
		"""This function gets a user's data field and info"""

		call = "/api/users/"+ str(email)

		return self.api_call(call=call, method="GET")