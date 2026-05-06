def render(self, request, instance, **kwargs):
        """
        The rendering/view function that displays a plugin model instance.

        :param instance: An instance of the ``model`` the plugin uses.
        :param request: The Django :class:`~django.http.HttpRequest` class containing the request parameters.
        :param kwargs: An optional slot for any new parameters.

        To render a plugin, either override this function, or specify the :attr:`render_template` variable,
        and optionally override :func:`get_context`.
        It is recommended to wrap the output in a ``<div>`` tag,
        to prevent the item from being displayed right next to the previous plugin.

        .. versionadded:: 1.0
           The function may either return a string of HTML code,
           or return a :class:`~fluent_contents.models.ContentItemOutput` object
           which holds both the CSS/JS includes and HTML string.
           For the sake of convenience and simplicity, most examples
           only return a HTML string directly.

           When the user needs to be redirected, simply return a :class:`~django.http.HttpResponseRedirect`
           or call the :func:`redirect` method.

        To render raw HTML code, use :func:`~django.utils.safestring.mark_safe` on the returned HTML.
        """
        render_template = self.get_render_template(request, instance, **kwargs)
        if not render_template:
            return str(_(u"{No rendering defined for class '%s'}" % self.__class__.__name__))

        context = self.get_context(request, instance, **kwargs)
        return self.render_to_string(request, render_template, context)