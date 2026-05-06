def render_to_response(self, context, **response_kwargs):
        """
        Overloaded to deal with _format arguments.
        """
        # is this a select2 format response?
        if self.request.GET.get('_format', 'html') == 'select2':

            results = []
            for obj in context['object_list']:
                result = None
                if hasattr(obj, 'as_select2'):
                    result = obj.as_select2()

                if not result:
                    result = dict(id=obj.pk, text="%s" % obj)

                results.append(result)

            json_data = dict(results=results, err='nil', more=context['page_obj'].has_next())
            return JsonResponse(json_data)
        # otherwise, return normally
        else:
            return super(SmartListView, self).render_to_response(context)