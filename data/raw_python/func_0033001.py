def render_search(self, ctx, data):
        """
        Render some UI for performing searches, if we know about a search
        aggregator.
        """
        if self.username is None:
            return ''
        translator = self._getViewerPrivateApplication()
        searchAggregator = translator.getPageComponents().searchAggregator
        if searchAggregator is None or not searchAggregator.providers():
            return ''
        return ctx.tag.fillSlots(
            'form-action', translator.linkTo(searchAggregator.storeID))