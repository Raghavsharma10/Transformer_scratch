def save_model(self, request, obj, form, change):
        """Updates all metrics with the same name"""
        like_metrics = self.model.objects.filter(name=obj.name)
        # 2.7+ only :(
        # = {key: form.cleaned_data[key] for key in form.changed_data}
        updates = {}
        for key in form.changed_data:
            updates[key] = form.cleaned_data[key]
        like_metrics.update(**updates)