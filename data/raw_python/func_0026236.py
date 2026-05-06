def form_valid(self, form):
        """ Call `form.save()` and super itself. """
        form.save()
        return super(SubscriptionView, self).form_valid(form)