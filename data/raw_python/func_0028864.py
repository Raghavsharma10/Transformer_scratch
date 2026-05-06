def render(self, indent=0):
        """
        Renders a HttpResponse for the ongoing request
        :param indent int
        :rtype: HttpResponse
        """
        self.__indent = indent
        return HttpResponse(
            str(self), content_type=self.__content_type, charset=self.__charset, **self.__kwargs
        )