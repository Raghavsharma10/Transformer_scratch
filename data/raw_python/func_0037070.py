def delete_user_by_email(self, email):
	
		"""
		This call will delete a user from the Iterable database.  
		This call requires a path parameter to be passed in, 'email'
		in this case, which is why we're just adding this to the 'call'
		argument that goes into the 'api_call' request. 		
		"""
		call = "/api/users/"+ str(email)

		return self.api_call(call=call, method="DELETE")