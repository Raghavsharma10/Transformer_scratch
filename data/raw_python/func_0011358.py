def finish_experiment(self, session, exp_id):
        '''remove an experiment from the list after completion.
        '''
        self.logger.debug('Finishing %s' %exp_id)
        experiments = session.get('experiments', [])
        experiments = [x for x in experiments if x != exp_id]
        session['experiments'] = experiments
        return experiments