def generate_user(self):
    '''generate a new user in the database, still session based so we
       create a new identifier. This function is called from the users new 
       entrypoint, and it assumes we want a user generated with a token.
    '''
    token = str(uuid.uuid4())
    return self.generate_subid(token=token, return_user=True)