def filterAllOr(self, **kwargs):
        '''
            filterAllOr - Perform a filter operation on ALL nodes in this collection and all their children.

            Results must match ANY the filter criteria. for ALL, use the *And methods

            For just the nodes in this collection, use "filterOr" on a TagCollection

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

        return filterableNodes.filterOr(**kwargs)