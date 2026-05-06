def _produceIt(self, segments, thunk):
        """
        Underlying implmeentation of L{PrefixURLMixin.produceResource} and
        L{PrefixURLMixin.sessionlessProduceResource}.

        @param segments: the URL segments to dispatch.

        @param thunk: a 0-argument callable which returns an L{IResource}
        provider, or None.

        @return: a 2-tuple of C{(resource, remainingSegments)}, or L{None}.
        """
        if not self.prefixURL:
            needle = ()
        else:
            needle = tuple(self.prefixURL.split('/'))
        S = len(needle)
        if segments[:S] == needle:
            if segments == JUST_SLASH:
                # I *HATE* THE WEB
                subsegments = segments
            else:
                subsegments = segments[S:]
            res = thunk()
            # Even though the URL matched up, sometimes we might still
            # decide to not handle this request (eg, some prerequisite
            # for our function is not met by the store).  Allow None
            # to be returned by createResource to indicate this case.
            if res is not None:
                return res, subsegments