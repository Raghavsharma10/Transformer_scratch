def save_data(self,session, exp_id, content):
    '''save data will obtain the current subid from the session, and save it
       depending on the database type.'''
    from expfactory.database.models import (
        Participant,
        Result
    )
    subid = session.get('subid') 
    bot.info('Saving data for subid %s' % subid)    

    # We only attempt save if there is a subject id, set at start
    if subid is not None:
        p = Participant.query.filter(Participant.id == subid).first() # better query here

        # Preference is to save data under 'data', otherwise do all of it
        if "data" in content:
            content = content['data']

        if isinstance(content,dict):
            content = json.dumps(content)

        result = Result(data=content,
                        exp_id=exp_id,
                        participant_id=p.id) # check if changes from str/int
        self.session.add(result)
        p.results.append(result)
        self.session.commit()

        bot.info("Participant: %s" %p)
        bot.info("Result: %s" %result)