def get_placeholder_data_view(self, request, object_id):
        """
        Return the placeholder data as dictionary.
        This is used in the client for the "copy" functionality.
        """
        language = 'en'  #request.POST['language']
        with translation.override(language):  # Use generic solution here, don't assume django-parler is used now.
            obj = self.get_object(request, object_id)

        if obj is None:
            json = {'success': False, 'error': 'Page not found'}
            status = 404
        elif not self.has_change_permission(request, obj):
            json = {'success': False, 'error': 'No access to page'}
            status = 403
        else:
            # Fetch the forms that would be displayed,
            # return the data as serialized form data.
            status = 200
            json = {
                'success': True,
                'object_id': object_id,
                'language_code': language,
                'formset_forms': self._get_object_formset_data(request, obj),
            }

        return JsonResponse(json, status=status)