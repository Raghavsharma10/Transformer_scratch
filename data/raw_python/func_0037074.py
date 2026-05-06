def update_user(self, email=None, data_fields=None, user_id=None,
					prefer_userId= None, merge_nested_objects=None):

		"""
		The Iterable 'User Update' api updates a user profile with new data 
		fields. Missing fields are not deleted and new data is merged.

		The body of the request takes 4 keys:
			1. email-- in the form of a string -- used as the unique identifier by
				the Iterable database.
			2. data fields-- in the form of an object-- these are the additional attributes
			 of the user that we want to add or update
			3. userId- in the form of a string-- another field we can use as a lookup
				of the user. 
			4. mergeNestedObjects-- in the form of an object-- used to merge top level
				objects instead of overwriting. 
		"""

		call = "/api/users/update"
		
		payload = {}

		if email is not None:
			payload["email"] = str(email)

		if data_fields is not None:
			payload["dataFields"] = data_fields

		if user_id is not None:
			payload["userId"] = str(user_id)

		if prefer_userId is not None:
			payload["preferUserId"]= prefer_userId

		if merge_nested_objects is not None:
			payload["mergeNestedObjects"] = merge_nested_objects
		
		return self.api_call(call=call, method="POST", json=payload)