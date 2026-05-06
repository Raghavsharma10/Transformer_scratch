def renderHTTP(self, ctx):
        """
        Handle the password reset form.

        The following exchange describes the process:

            S: Render C{reset}
            C: POST C{username} or C{email}
            S: L{handleRequestForUser}, render C{reset-check-email}

            (User follows the emailed reset link)

            S: Render C{reset-step-two}
            C: POST C{password1}
            S: L{resetPassword}, render C{reset-done}
        """
        req = inevow.IRequest(ctx)

        if req.method == 'POST':
            if req.args.get('username', [''])[0]:
                user = unicode(usernameFromRequest(req), 'ascii')
                self.handleRequestForUser(user, URL.fromContext(ctx))
                self.fragment = self.templateResolver.getDocFactory(
                    'reset-check-email')
            elif req.args.get('email', [''])[0]:
                email = req.args['email'][0].decode('ascii')
                acct = self.accountByAddress(email)
                if acct is not None:
                    username = '@'.join(
                        userbase.getAccountNames(acct.avatars.open()).next())
                    self.handleRequestForUser(username, URL.fromContext(ctx))
                self.fragment = self.templateResolver.getDocFactory('reset-check-email')
            elif 'password1' in req.args:
                (password,) = req.args['password1']
                self.resetPassword(self.attempt, unicode(password))
                self.fragment = self.templateResolver.getDocFactory('reset-done')
            else:
                # Empty submit;  redirect back to self
                return URL.fromContext(ctx)
        elif self.attempt:
            self.fragment = self.templateResolver.getDocFactory('reset-step-two')

        return PublicPage.renderHTTP(self, ctx)