def validate_token(self, token):
    '''retrieve a subject based on a token. Valid means we return a participant
       invalid means we return None
    '''
    # A token that is finished or revoked is not valid
    subid = None
    if not token.endswith(('finished','revoked')):
        subid = self.generate_subid(token=token)
        data_base = "%s/%s" %(self.data_base, subid)
        if not os.path.exists(data_base):
            subid = None
    return subid