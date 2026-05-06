def delete_translations_for_item_and_its_children(self, item, languages=None):
        """
        deletes the translations task of an item and its children
        used when a model is not enabled anymore
        :param item:
        :param languages:
        :return:
        """

        self.log('--- Deleting translations ---')

        if not self.master:
            self.set_master(item)

        object_name = '{} - {}'.format(item._meta.app_label.lower(), item._meta.verbose_name)
        object_class = item.__class__.__name__
        object_pk = item.pk

        filter_by = {
            'object_class': object_class,
            'object_name': object_name,
            'object_pk': object_pk,
            'done': False
        }
        if languages:
            filter_by.update({'language__code__in': languages})
        TransTask.objects.filter(**filter_by).delete()

        # then process child objects from main
        children = self.get_translatable_children(item)
        for child in children:
            self.delete_translations_for_item_and_its_children(child, languages)