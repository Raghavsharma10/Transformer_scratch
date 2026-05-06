def form_valid(self, form):
        """Get the category name/metric slugs from the form, and update the
        category so contains the given metrics."""
        form.categorize_metrics()
        return super(CategoryFormView, self).form_valid(form)