def post_save(self, obj, created=False):
        """indexes the object to ElasticSearch after any save function (POST/PUT)

        :param obj: instance of the saved object
        :param created: boolean expressing if object is newly created (`False` if updated)
        :return: `rest_framework.viewset.ModelViewSet.post_save`
        """
        from bulbs.content.tasks import index

        index.delay(obj.polymorphic_ctype_id, obj.pk)

        message = "Created" if created else "Saved"
        LogEntry.objects.log(self.request.user, obj, message)
        return super(ContentViewSet, self).post_save(obj, created=created)