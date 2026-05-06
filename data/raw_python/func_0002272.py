def remove_stale_items(self, stale_cts):
        """
        See if there are items that point to a removed model.
        """
        stale_ct_ids = list(stale_cts.keys())
        items = (ContentItem.objects
                 .non_polymorphic()  # very important, or polymorphic skips them on fetching derived data
                 .filter(polymorphic_ctype__in=stale_ct_ids)
                 .order_by('polymorphic_ctype', 'pk')
                 )
        if not items:
            self.stdout.write("No stale items found.")
            return

        if self.dry_run:
            self.stdout.write("The following content items are stale:")
        else:
            self.stdout.write("The following content items were stale:")

        for item in items:
            ct = stale_cts[item.polymorphic_ctype_id]
            self.stdout.write("- #{id} points to removed {app_label}.{model}".format(
                id=item.pk, app_label=ct.app_label, model=ct.model
            ))

            if not self.dry_run:
                try:
                    item.delete()
                except PluginNotFound:
                    Model.delete(item)