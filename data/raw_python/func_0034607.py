def boto_fix_security_token_in_profile(self, connect_args):
        ''' monkey patch for boto issue boto/boto#2100 '''
        profile = 'profile ' + self.boto_profile
        if boto.config.has_option(profile, 'aws_security_token'):
            connect_args['security_token'] = boto.config.get(profile, 'aws_security_token')
        return connect_args