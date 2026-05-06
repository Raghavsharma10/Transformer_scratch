def get_render_data(self, **kwargs):
        """
        Adds the model_name to the context, then calls super.
        """
        kwargs['model_name'] = self.model_name
        kwargs['model_name_plural'] = self.model_name_plural
        return super(ModelCMSView, self).get_render_data(**kwargs)