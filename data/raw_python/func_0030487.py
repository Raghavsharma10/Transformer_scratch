def handle_deletions(self):
        """
        Manages handling deletions of objects that were previously managed by the initial data process but no longer
        managed. It does so by mantaining a list of receipts for model objects that are registered for deletion on
        each round of initial data processing. Any receipts that are from previous rounds and not the current
        round will be deleted.
        """

        deduplicated_objs = {}
        for model in self.model_objs_registered_for_deletion:
            key = '{0}:{1}'.format(
                ContentType.objects.get_for_model(model, for_concrete_model=False),
                model.id
            )
            deduplicated_objs[key] = model

        # Create receipts for every object registered for deletion
        now = timezone.now()
        registered_for_deletion_receipts = [
            RegisteredForDeletionReceipt(
                model_obj_type=ContentType.objects.get_for_model(model_obj, for_concrete_model=False),
                model_obj_id=model_obj.id,
                register_time=now)
            for model_obj in deduplicated_objs.values()
        ]

        # Do a bulk upsert on all of the receipts, updating their registration time.
        RegisteredForDeletionReceipt.objects.bulk_upsert(
            registered_for_deletion_receipts, ['model_obj_type_id', 'model_obj_id'], update_fields=['register_time'])

        # Delete all receipts and their associated model objects that weren't updated
        for receipt in RegisteredForDeletionReceipt.objects.exclude(register_time=now):
            try:
                receipt.model_obj.delete()
            except:  # noqa
                # The model object may no longer be there, its ctype may be invalid, or it might be protected.
                # Regardless, the model object cannot be deleted, so go ahead and delete its receipt.
                pass
            receipt.delete()