def get_renderers(self):
        """
        Instantiates and returns the list of renderers that this view can use.
        """
        try:
            source = self.get_object()
        except (ImproperlyConfigured, APIException):
            self.renderer_classes = [RENDERER_MAPPING[i] for i in self.__class__.renderers]
            return [RENDERER_MAPPING[i]() for i in self.__class__.renderers]
        else:
            self.renderer_classes = [RENDERER_MAPPING[i] for i in source.__class__.renderers]
            return [RENDERER_MAPPING[i]() for i in source.__class__.renderers]