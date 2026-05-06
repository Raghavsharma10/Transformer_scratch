def kw_changelist_view(self, request: HttpRequest, extra_context=None, **kw):
        """
        Changelist view which allow key-value arguments.
        :param request: HttpRequest
        :param extra_context: Extra context dict
        :param kw: Key-value dict
        :return: See changelist_view()
        """
        return self.changelist_view(request, extra_context)