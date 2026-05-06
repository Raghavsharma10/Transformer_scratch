def container_name(self):
        """
        The container_name is the concatenation of ``image_name`` and a uuid1 string

        We also remove the url portion of the ``image_name`` before using it.
        """
        if getattr(self, "_container_name", NotSpecified) is NotSpecified:
            self.container_name = "{0}-{1}".format(self.image_name.replace("/", "--").replace(":", "---"), str(uuid.uuid1()).lower())
        return self._container_name