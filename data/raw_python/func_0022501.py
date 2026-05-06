def categorize_metrics(self):
        """Called only on a valid form, this method will place the chosen
        metrics in the given catgory."""
        category = self.cleaned_data['category_name']
        metrics = self.cleaned_data['metrics']
        self.r.reset_category(category, metrics)