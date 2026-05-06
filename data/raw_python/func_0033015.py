def render_head(self, ctx, data):
        """
        Put liveglue content into the header of this page to activate it, but
        otherwise delegate to my parent's renderer for <head>.
        """
        ctx.tag[tags.invisible(render=tags.directive('liveglue'))]
        return _PublicPageMixin.render_head(self, ctx, data)