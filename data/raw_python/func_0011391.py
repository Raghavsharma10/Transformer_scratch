def validate(self, url):
        ''' takes in a Github repository for validation of preview and 
            runtime (and possibly tests passing?
        '''

        # Preview must provide the live URL of the repository
        if not url.startswith('http') or not 'github' in url:
            bot.error('Test of preview must be given a Github repostitory.')
            return False

        if not self._validate_preview(url):
            return False

        return True