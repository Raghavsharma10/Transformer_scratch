def remove_unreferenced_items(self, stale_cts):
        """
        See if there are items that no longer point to an existing parent.
        """
        stale_ct_ids = list(stale_cts.keys())
        parent_types = (ContentItem.objects.order_by()
                        .exclude(polymorphic_ctype__in=stale_ct_ids)
                        .values_list('parent_type', flat=True).distinct())

        num_unreferenced = 0

        for ct_id in parent_types:
            parent_ct = ContentType.objects.get_for_id(ct_id)
            unreferenced_items = (ContentItem.objects
                                  .filter(parent_type=ct_id)
                                  .order_by('polymorphic_ctype', 'pk'))

            if parent_ct.model_class() is not None:
                # Only select the items that are part of removed pages,
                # unless the parent type was removed - then removing all is correct.
                unreferenced_items = unreferenced_items.exclude(
                    parent_id__in=parent_ct.get_all_objects_for_this_type()
                )

            if unreferenced_items:
                for item in unreferenced_items:
                    self.stdout.write(
                        "- {cls}#{id} points to nonexisting {app_label}.{model}".format(
                            cls=item.__class__.__name__, id=item.pk,
                            app_label=parent_ct.app_label, model=parent_ct.model
                        ))
                    num_unreferenced += 1
                    if not self.dry_run and self.remove_unreferenced:
                        item.delete()

        if not num_unreferenced:
            self.stdout.write("No unreferenced items found.")
        else:
            self.stdout.write("{0} unreferenced items found.".format(num_unreferenced))
            if not self.remove_unreferenced:
                self.stdout.write("Re-run this command with --remove-unreferenced to remove these items")