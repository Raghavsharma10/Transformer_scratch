def filterAll(self, **kwargs):
        '''
            filterAll aka filterAllAnd - Perform a filter operation on ALL nodes in this collection and all their children.

            Results must match ALL the filter criteria. for ANY, use the *Or methods

            For just the nodes in this collection, use "filter" or "filterAnd" on a TagCollection

            For special filter keys, @see #AdvancedHTMLParser.AdvancedHTMLParser.filter

            Requires the QueryableList module to be installed (i.e. AdvancedHTMLParser was installed
              without '--no-deps' flag.)
            
            For alternative without QueryableList,
              consider #AdvancedHTMLParser.AdvancedHTMLParser.find method or the getElement* methods

            @return TagCollection<AdvancedTag>
        '''
        if canFilterTags is False:
            raise NotImplementedError('filter methods requires QueryableList installed, it is not. Either install QueryableList, or try the less-robust "find" method, or the getElement* methods.')

        allNodes = self.getAllNodes()

        filterableNodes = FilterableTagCollection(allNodes)

        return filterableNodes.filterAnd(**kwargs)