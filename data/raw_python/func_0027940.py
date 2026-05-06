def full_restore(self, using=None):
        using = using or router.db_for_write(self.__class__, instance=self)
        restore_counter = Counter()
        if self.deleted:
            time = self.deleted_at
        else:
            return restore_counter
        self.collector = models.deletion.Collector(using=using)
        self.collector.collect([self])
        

        for model, instances in self.collector.data.items():
            instances_to_delete = sorted(instances, key=attrgetter("pk"))

        self.sort()

        for qs in self.collector.fast_deletes:
            # TODO make sure the queryset delete has been made a soft delete
            for qs_instance in qs:
                restore_counter.update([qs_instance._meta.model_name])
                qs_instance.restore(time=time)

        for model, instances in self.collector.data.items():
            for instance in instances:
                restore_counter.update([instance._meta.model_name])
                instance.restore(time=time)

        return sum(restore_counter.values()), dict(restore_counter)

        """
        Restores itself, as well as objects that might have been deleted along with it if cascade is the deletion strategy
        """
        self.collector = models.deletion.Collector(using=using) 
        self.collector.collect([self], keep_parents=keep_parents)