def render_to_response(self, context, **response_kwargs):
        """
        Overloaded to deal with _format arguments.
        """
        # should we actually render in json?
        if '_format' in self.request.GET and self.request.GET['_format'] == 'json':
            return JsonResponse(self.as_json(context), safe=False)

        # otherwise, return normally
        else:
            return super(SmartView, self).render_to_response(context)