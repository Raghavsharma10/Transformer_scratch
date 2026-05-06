def build_subreverse(self, _name, **kwargs):
        '''
        String-based reverse API. Returns subreverse object::

            env.root.build_subreverse('user', user_id=1).profile
        '''
        _, subreverse = self._build_url_silent(_name, **kwargs)
        return subreverse