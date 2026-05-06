def render_urchin(self, ctx, data):
        """
        Render the code for recording Google Analytics statistics, if so
        configured.
        """
        key = website.APIKey.getKeyForAPI(self.store, website.APIKey.URCHIN)
        if key is None:
            return ''
        return ctx.tag.fillSlots('urchin-key', key.apiKey)