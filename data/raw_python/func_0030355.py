def logout(self, redirect_to='/'):
        '''
        This property will return component which will handle logout requests.
        It only handles POST requests and do not display any rendered content.
        This handler deletes session id from `storage`. If there is no
        session id provided or id is incorrect handler silently redirects to
        login url and does not throw any exception.
        '''
        def _logout(env, data):
            location = redirect_to
            if location is None and env.request.referer:
                location = env.request.referer
            elif location is None:
                location = '/'
            response = HTTPSeeOther(location=str(location))
            self.logout_user(env.request, response)
            return response
        return web.match('/logout', 'logout') | web.method('post') | _logout