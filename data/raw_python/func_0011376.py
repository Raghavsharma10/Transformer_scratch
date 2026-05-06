def generate_user(self, subid=None):
    '''generate a new user on the filesystem, still session based so we
       create a new identifier. This function is called from the users new 
       entrypoint, and it assumes we want a user generated with a token.
       since we don't have a database proper, we write the folder name to 
       the filesystem
    '''
    # Only generate token if subid being created
    if subid is None:
        token = str(uuid.uuid4())
        subid = self.generate_subid(token=token)

    if os.path.exists(self.data_base):    # /scif/data
        data_base = "%s/%s" %(self.data_base, subid)
        # expfactory/00001
        if not os.path.exists(data_base):
            mkdir_p(data_base)

    return data_base