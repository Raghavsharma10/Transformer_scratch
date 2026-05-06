def perform_update(self, serializer):
        """creates a record in the `bulbs.promotion.PZoneHistory`

        :param obj: the instance saved
        :param created: boolean expressing if the object was newly created (`False` if updated)
        """
        instance = serializer.save()
        # create history object
        instance.history.create(data=instance.data)