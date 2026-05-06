def list_users(self):
    '''list users, each associated with a filesystem folder
    ''' 
    folders = glob('%s/*' %(self.database))
    folders.sort()
    return [self.print_user(x) for x in folders]