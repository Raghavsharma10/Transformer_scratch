def run(self, serialized_instance, *args, **kwargs):
        """ Deserialize input data and start backend operation execution """
        try:
            instance = utils.deserialize_instance(serialized_instance)
        except ObjectDoesNotExist:
            message = ('Cannot restore instance from serialized object %s. Probably it was deleted.' %
                       serialized_instance)
            six.reraise(ObjectDoesNotExist, message)

        self.args = args
        self.kwargs = kwargs

        self.pre_execute(instance)
        result = self.execute(instance, *self.args, **self.kwargs)
        self.post_execute(instance)
        if result and isinstance(result, django_models.Model):
            result = utils.serialize_instance(result)
        return result