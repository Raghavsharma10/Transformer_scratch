def get_item(self, **kwargs):
        """ Reload context on each access. """
        self.reload_context(es_based=False, **kwargs)
        return super(ItemSubresourceBaseView, self).get_item(**kwargs)