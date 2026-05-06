def save_data(self,session, exp_id, content):
    '''save data will obtain the current subid from the session, and save it
       depending on the database type. Currently we just support flat files'''
    from expfactory.database.models import (
        Participant,
        Result
    )
    subid = session.get('subid')
    token = session.get('token') 

    self.logger.info('Saving data for subid %s' % subid)    

    # We only attempt save if there is a subject id, set at start
    if subid is not None:
        p = Participant.query.filter(Participant.id == subid).first() # better query here

        # Does 
        if self.headless and p.token != token:
            self.logger.warning('%s attempting to use mismatched token [%s] skipping save' %(p.id, token))
        elif self.headless and p.token.endswith(('finished','revoked')):
            self.logger.warning('%s attempting to use expired token [%s] skipping save' %(p.id, token))
        else:

            # Preference is to save data under 'data', otherwise do all of it
            if "data" in content:
                content = content['data']

            result = Result(data=content,
                            exp_id=exp_id,
                            participant_id=p.id) # check if changes from str/int

            # Create and save the result
            self.session.add(result)
            p.results.append(result)
            self.session.commit()

            self.logger.info("Save [participant] %s [result] %s" %(p, result))