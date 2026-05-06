def mount(self, route: str, controller: callable) -> url:
        """
        Maps a route namespace with the given params and point it's requests to the especified controller.
        :param route: str Namespace route to be mapped
        :param controller: callback Controller callable to map end-points
        :rtype: url
        """
        if issubclass(controller, TemplateView):
            return url(
                r"%s" % route,
                Router(self, route, controller).handle
            )
        else:
            raise TypeError("The controller %s must be a subclass of %s" % (
                    controller, TemplateView
                )
            )