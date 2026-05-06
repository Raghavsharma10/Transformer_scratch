def bulk_update_user(self, users):

		"""
		The Iterable 'Bulk User Update' api Bulk update user data or adds 
		it if does not exist. Data is merged - missing fields are not deleted

		The body of the request takes 1 keys:
			1. users -- in the form of an array -- which is the list of users
				that we're updating in sets of 50 users at a time, which is the 
				most that can be batched in a single request.  
		"""

		call = "/api/users/bulkUpdate"		
		
		payload = {}

		if isinstance(users, list):
			payload["users"] = users
		else:
			raise TypeError ('users are not in Arrary format')		

		return self.api_call(call=call, method="POST", json=payload)