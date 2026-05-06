def finish_user(self, subid, ext='finished'):
    '''finish user will append "finished" (or other) to the data folder when
       the user has completed (or been revoked from) the battery. 
       For headless, this means that the session is ended and the token 
       will not work again to rewrite the result. If the user needs to update
       or redo an experiment, this can be done with a new session. Note that if
       this function is called internally by the application at experiment
       finish, the subid includes a study id (e.g., expfactory/xxxx-xxxx)
       but if called by the user, it may not (e.g., xxxx-xxxx). We check
       for this to ensure it works in both places.
    '''
    if os.path.exists(self.data_base):    # /scif/data

        # Only relevant to filesystem save - the studyid is the top folder
        if subid.startswith(self.study_id):
            data_base = "%s/%s" %(self.data_base, subid)
        else:
            data_base = "%s/%s/%s" %(self.data_base,
                                     self.study_id,
                                     subid)

        # The renamed file will be here
        finished = "%s_%s" % (data_base, ext)

        # Participant already finished
        if os.path.exists(finished):
            self.logger.warning('[%s] is already finished: %s' % (subid, data_base))

        # Exists and can finish
        elif os.path.exists(data_base):
            os.rename(data_base, finished)

        # Not finished, doesn't exist
        else:
            finished = None
            self.logger.warning('%s does not exist, cannot finish. %s' % (data_base, subid))

    return finished