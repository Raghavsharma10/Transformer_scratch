def save_model(self, request, obj, form, change):
        """Add an ObjectPosition to the object."""
        super(GenericPositionsAdmin, self).save_model(request, obj, form,
                                                      change)
        c_type = ContentType.objects.get_for_model(obj)
        try:
            ObjectPosition.objects.get(content_type__pk=c_type.id,
                                       object_id=obj.id)
        except ObjectPosition.DoesNotExist:
            position_objects = ObjectPosition.objects.filter(
                content_type__pk=c_type.id, position__isnull=False).order_by(
                    '-position')
            try:
                position = (position_objects[0].position + 1)
            except IndexError:
                position = 1
            ObjectPosition.objects.create(
                content_object=obj, position=position)