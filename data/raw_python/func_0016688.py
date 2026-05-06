def _delete_images(self, instance):
        """Deletes all user media images of the given instance."""
        UserMediaImage.objects.filter(
            content_type=ContentType.objects.get_for_model(instance),
            object_id=instance.pk,
            user=instance.user,
        ).delete()