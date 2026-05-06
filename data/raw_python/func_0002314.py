def get_layout_view(self, request):
        """
        Return the metadata about a layout
        """
        template_name = request.GET['name']

        # Check if template is allowed, avoid parsing random templates
        templates = dict(appconfig.SIMPLECMS_TEMPLATE_CHOICES)
        if template_name not in templates:
            jsondata = {'success': False, 'error': 'Template not found'}
            status = 404
        else:
            # Extract placeholders from the template, and pass to the client.
            template = get_template(template_name)
            placeholders = get_template_placeholder_data(template)

            jsondata = {
                'placeholders': [p.as_dict() for p in placeholders],
            }
            status = 200

        jsonstr = json.dumps(jsondata)
        return HttpResponse(jsonstr, content_type='application/json', status=status)