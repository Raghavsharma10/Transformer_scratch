def list_drafts(self):
        """
        A filterable list views of layers, returning the draft version of each layer.
        If the most recent version of a layer or table has been published already,
        it won’t be returned here.
        """
        target_url = self.client.get_url('LAYER', 'GET', 'multidraft')
        return base.Query(self, target_url)