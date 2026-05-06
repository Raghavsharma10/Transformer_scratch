def get_next(self, session):
        '''return the name of the next experiment, depending on the user's
           choice to randomize. We don't remove any experiments here, that is
           done on finish, in the case the user doesn't submit data (and
           thus finish). A return of None means the user has completed the
           battery of experiments.
        '''
        next = None
        experiments = session.get('experiments', [])
        if len(experiments) > 0:    
            if app.randomize is True:
                next = random.choice(range(0,len(experiments)))
                next = experiments[next]
            else:
                next = experiments[0]
        return next