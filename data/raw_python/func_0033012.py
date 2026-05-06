def beforeRender(self, ctx):
        """
        Before rendering this page, identify the correct URL for the login to post
        to, and the error message to display (if any), and fill the 'login
        action' and 'error' slots in the template accordingly.
        """
        generator = ixmantissa.ISiteURLGenerator(self.store)
        url = generator.rootURL(IRequest(ctx))
        url = url.child('__login__')
        for seg in self.segments:
            url = url.child(seg)
        for queryKey, queryValues in self.arguments.iteritems():
            for queryValue in queryValues:
                url = url.add(queryKey, queryValue)

        req = inevow.IRequest(ctx)
        err = req.args.get('login-failure', ('',))[0]

        if 0 < len(err):
            error = inevow.IQ(
                        self.fragment).onePattern(
                                'error').fillSlots('error', err)
        else:
            error = ''

        ctx.fillSlots("login-action", url)
        ctx.fillSlots("error", error)