def refresh_token(self, subid):
    '''refresh or generate a new token for a user. If the user is finished,
       this will also make the folder available again for using.'''
    if os.path.exists(self.data_base):    # /scif/data
        data_base = "%s/%s" %(self.data_base, subid)
        if os.path.exists(data_base):
            refreshed = "%s/%s" %(self.database, str(uuid.uuid4()))
            os.rename(data_base, refreshed)
            return refreshed
        self.logger.warning('%s does not exist, cannot rename %s' % (data_base, subid))
    else:
        self.logger.warning('%s does not exist, cannot rename %s' % (self.database, subid))