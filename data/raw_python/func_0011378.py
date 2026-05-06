def restart_user(self, subid):
    '''restart user will remove any "finished" or "revoked" extensions from 
    the user folder to restart the session. This command always comes from
    the client users function, so we know subid does not start with the
    study identifer first
    '''        
    if os.path.exists(self.data_base): # /scif/data/<study_id>
        data_base = "%s/%s" %(self.data_base, subid)
        for ext in ['revoked','finished']:
            folder = "%s_%s" % (data_base, ext)
            if os.path.exists(folder):
                os.rename(folder, data_base)
                self.logger.info('Restarting %s, folder is %s.' % (subid, data_base))

        self.logger.warning('%s does not have revoked or finished folder, no changes necessary.' % (subid))
        return data_base    

    self.logger.warning('%s does not exist, cannot restart. %s' % (self.database, subid))