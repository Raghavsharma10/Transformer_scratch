def login(self, template='login'):
        '''
        This property will return component which will handle login requests.

            auth.login(template='login.html')
        '''
        def _login(env, data):
            form = self._login_form(env)
            next = env.request.GET.get('next', '/')
            login_failed = False
            if env.request.method == 'POST':
                if form.accept(env.request.POST):
                    user_identity = self.get_user_identity(
                                                env, **form.python_data)
                    if user_identity is not None:
                        response = HTTPSeeOther(location=next)
                        return self.login_identity(user_identity, response)
                    login_failed = True
            data.form = form
            data.login_failed = login_failed
            data.login_url = env.root.login.as_url.qs_set(next=next)
            return env.template.render_to_response(template, data.as_dict())
        return web.match('/login', 'login') | _login