def form_valid(self, form):
        """Pull the metrics from the submitted form, and store them as a
        list of strings in ``self.metric_slugs``.
        """
        self.metric_slugs = [k.strip() for k in form.cleaned_data['metrics']]
        return super(AggregateFormView, self).form_valid(form)