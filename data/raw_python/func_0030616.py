def build_url(self, _name, **kwargs):
        '''
        String-based reverse API. Returns URL object::

            env.root.build_url('user.profile', user_id=1)

        Checks that all necessary arguments are provided and all
        provided arguments are used.
        '''
        used_args, subreverse =  self._build_url_silent(_name, **kwargs)

        if set(kwargs).difference(used_args):
            raise UrlBuildingError(
                'Not all arguments are used during URL building: {}'\
                    .format(', '.join(set(kwargs).difference(used_args))))
        return subreverse.as_url