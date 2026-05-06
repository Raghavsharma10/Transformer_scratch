def render_authenticateLinks(self, ctx, data):
        """
        For unauthenticated users, add login and signup links to the given tag.
        For authenticated users, remove the given tag from the output.

        When necessary, the I{signup-link} pattern will be loaded from the tag.
        Each copy of it will have I{prompt} and I{url} slots filled.  The list
        of copies will be added as children of the tag.
        """
        if self.username is not None:
            return ''
        # there is a circular import here which should probably be avoidable,
        # since we don't actually need signup links on the signup page.  on the
        # other hand, maybe we want to eventually put those there for
        # consistency.  for now, this import is easiest, and although it's a
        # "friend" API, which I dislike, it doesn't seem to cause any real
        # problems...  -glyph
        from xmantissa.signup import _getPublicSignupInfo

        IQ = inevow.IQ(ctx.tag)
        signupPattern = IQ.patternGenerator('signup-link')

        signups = []
        for (prompt, url) in _getPublicSignupInfo(self.store):
            signups.append(signupPattern.fillSlots(
                    'prompt', prompt).fillSlots(
                    'url', url))

        return ctx.tag[signups]