def print_user(self, user):
    '''print a filesystem database user. A "database" folder that might end with
       the participant status (e.g. _finished) is extracted to print in format
 
       [folder]                        [identifier][studyid]
       /scif/data/expfactory/xxxx-xxxx   xxxx-xxxx[studyid]
       
    ''' 
    status = "active"

    if user.endswith('_finished'):
        status = "finished"

    elif user.endswith('_revoked'):
        status = "revoked"

    subid = os.path.basename(user)
    for ext in ['_revoked','_finished']:
        subid = subid.replace(ext, '')
  
    subid = '%s\t%s[%s]' %(user, subid, status)
    print(subid)
    return subid