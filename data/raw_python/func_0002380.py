def rebuild_tree(self, request):
        '''
        Rebuilds the tree and clears the cache.
        '''
        self.model.objects.rebuild()
        self.message_user(request, _('Menu Tree Rebuilt.'))
        return self.clean_cache(request)