def locateChild(self, context, segments):
        """
        Return a statically defined child or a child defined by a sessionless
        site root plugin or an avatar from guard.
        """
        shortcut = getattr(self, 'child_' + segments[0], None)
        if shortcut:
            res = shortcut(context)
            if res is not None:
                return res, segments[1:]

        req = IRequest(context)
        for plg in self.siteStore.powerupsFor(ISessionlessSiteRootPlugin):
            spr = getattr(plg, 'sessionlessProduceResource', None)
            if spr is not None:
                childAndSegments = spr(req, segments)
            else:
                childAndSegments = plg.resourceFactory(segments)
            if childAndSegments is not None:
                return childAndSegments

        return self.guardedRoot.locateChild(context, segments)