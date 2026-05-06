def get_templates(self, action='index'):
        """
        Utility function that provides a list of templates to try for a given
        view, when the template isn't overridden by one of the template
        attributes on the class.
        """
        app = self.opts.app_label
        model_name = self.opts.model_name
        return [
            'wagtailmodeladmin/%s/%s/%s.html' % (app, model_name, action),
            'wagtailmodeladmin/%s/%s.html' % (app, action),
            'wagtailmodeladmin/%s.html' % (action,),
        ]