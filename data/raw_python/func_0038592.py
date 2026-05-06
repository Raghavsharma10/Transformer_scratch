def register_serializers(self, serializers):
        """
        Adds extra serializers; generally registered during the handler lifecycle
        """
        for new_serializer in serializers:

            if not isinstance(new_serializer, serializer.Base):
                msg = "registered serializer %s.%s does not inherit from prestans.serializer.Serializer" % (
                    new_serializer.__module__,
                    new_serializer.__class__.__name__
                )
                raise TypeError(msg)

        self._serializers = self._serializers + serializers