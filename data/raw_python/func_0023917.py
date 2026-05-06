def get_context_data(self, **kwargs):
        '''
        Set a base context
        '''

        # Call the base implementation first to get a context
        context = super(GenBase, self).get_context_data(**kwargs)

        # Update general context with the stuff we already calculated
        if hasattr(self, 'html_head'):
            context['html_head'] = self.html_head(self.object)

        # Add translation system
        if hasattr(self, 'gentrans'):
            context['gentranslate'] = self.gentrans.copy()
            context['gentranslate'].update(self.gentranslate)
        else:
            context['gentranslate'] = self.gentranslate

        # Return context
        return context