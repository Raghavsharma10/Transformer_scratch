def get_template_names(self):
        """
        Returns a list of template names for the view.

        :rtype: list.
        """
        #noinspection PyUnresolvedReferences
        if self.request.is_ajax():
            template_name = '/results.html'
        else:
            template_name = '/index.html'

        return ['{0}{1}'.format(self.template_dir, template_name)]