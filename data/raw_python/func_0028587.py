def login(self):
        """ perform API auth test returning user and team """
        log.debug('performing auth test')
        test = self._get(urls['test'])
        user = User({ 'name': test['user'], 'id': test['user_id'] })
        self._refresh()
        return test['team'], user