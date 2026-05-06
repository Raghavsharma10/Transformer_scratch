def get_form_class(self):
        """
        Returns the form class to use in this view
        """
        if self.form_class:
            form_class = self.form_class

        else:
            if self.model is not None:
                # If a model has been explicitly provided, use it
                model = self.model
            elif hasattr(self, 'object') and self.object is not None:
                # If this view is operating on a single object, use
                # the class of that object
                model = self.object.__class__
            else:
                # Try to get a queryset and extract the model class
                # from that
                model = self.get_queryset().model

            # run time parameters when building our form
            factory_kwargs = self.get_factory_kwargs()
            form_class = model_forms.modelform_factory(model, **factory_kwargs)

        return form_class