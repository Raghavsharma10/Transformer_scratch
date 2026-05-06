def print_user(self, user):
    '''print a relational database user
    ''' 
    status = "active"
    token = user.token

    if token in ['finished', 'revoked']:
        status = token

    if token is None:
        token = ''
  
    subid = "%s\t%s[%s]" %(user.id, token, status)
    print(subid)
    return subid