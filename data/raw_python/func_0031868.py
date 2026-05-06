def fake_m2m(self, obj, field_name):
        """
        Return the random objects from m2m relationship.
        The ManyToManyField need specific object,
        so i handle it after created the object.
        """
        instance_m2m = getattr(obj, field_name)
        objects_m2m = instance_m2m.model.objects.all()

        if objects_m2m.exists():
            ids_m2m = [i.pk for i in objects_m2m]
            random_decission = random.sample(
                range(min(ids_m2m), max(ids_m2m)), max(ids_m2m) - 1
            )
            if len(random_decission) <= 2:
                random_decission = [
                    self.djipsum_fields().randomize(ids_m2m)
                ]
            related_objects = [
                rel_obj for rel_obj in objects_m2m
                if rel_obj.pk in random_decission
            ]
            instance_m2m.add(*related_objects)