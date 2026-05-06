def get_search_form(self):
        """
        Returns search form instance.

        :rtype: django.forms.ModelForm.
        """
        #noinspection PyUnresolvedReferences
        if 'q' in self.request.GET:
            #noinspection PyUnresolvedReferences
            return self.search_form_class(self.request.GET)
        else:
            return self.search_form_class(placeholder=_(u'Search'))