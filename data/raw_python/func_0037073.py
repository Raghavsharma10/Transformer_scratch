def disable_device(self, token, email=None, user_id=None):
		"""
		This request manually disable pushes to a device until it comes
		online again.

		"""

		call = "/api/users/disableDevice"

		payload ={}

		payload["token"] = str(token)

		if email is not None:	
			payload["email"] = str(email)

		if user_id is not None:
			payload["userId"] = str(user_id)

		return self.api_call(call= call, method="POST", json=payload)