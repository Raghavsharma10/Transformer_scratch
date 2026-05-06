def create_from_item(self, languages, item, fields, trans_instance=None):
        """
        Creates tasks from a model instance "item" (master)
        Used in the api call

        :param languages:
        :param item:
        :param fields:
        :param trans_instance: determines if we are in bulk mode or not.
        If it has a value we're processing by the signal trigger,
        if not we're processing either by the api or the mixin
        :return:
        """
        if not isinstance(item, TranslatableModel):
            return

        self.log('gonna parse fields: {}'.format(fields))

        with self.lock:
            result = []

            if trans_instance:
                # get the values from the instance that is being saved, values not saved yet
                trans = trans_instance
            else:
                # get the values from the db instance
                trans = self.get_translation_from_instance(item, self.main_language)

            if not trans:
                return

            for field in fields:

                self.log('parsing field: {}'.format(field))

                # for every field
                object_field_label = self.get_field_label(trans, field)
                object_field_value = getattr(trans, field)
                # if object_field_value is None or object_field_value == '':
                #     object_field_value = getattr(self.instance, field, '')

                self.log('object_field_value for {} - .{}.'.format(object_field_label, object_field_value))

                if object_field_value == '' or object_field_value is None:
                    continue

                for lang in languages:
                    # for every language
                    self.log('parsing lang: {}'.format(lang))

                    language = TransLanguage.objects.filter(code=lang).get()
                    users = self.translators.get(lang, [])

                    self.log('gonna parse users')

                    for user in users:
                        # for every user we create a task

                        # check if there is already a value for the destinatation lang
                        # when we are in bulk mode, when we are in signal mode
                        # we update the destination task if it exists
                        if self.bulk_mode and self.exists_destination_lang_value(item, field, lang):
                            continue

                        ob_class_name = item.__class__.__name__

                        self.log('creating or updating object_class: {} | object_pk:{} | object_field: {}'.format(
                            ob_class_name,
                            item.pk,
                            field
                        ))

                        app_label = item._meta.app_label
                        model = ob_class_name.lower()
                        ct = ContentType.objects.get_by_natural_key(app_label, model)

                        try:
                            task, created = TransTask.objects.get_or_create(
                                content_type=ct,
                                object_class=ob_class_name,
                                object_pk=item.pk,
                                object_field=field,
                                language=language,
                                user=user,
                                defaults={
                                    'object_name': '{} - {}'.format(app_label, item._meta.verbose_name),
                                    'object_field_label': object_field_label,
                                    'object_field_value': object_field_value,
                                    'done': False
                                }
                            )
                            if not created:
                                self.log('updating')
                                task.date_modification = datetime.now()
                                task.object_field_value = object_field_value
                                task.done = False
                                task.save()
                            result.append(task)
                        except TransTask.MultipleObjectsReturned:
                            # theorically it should not occur but if so delete the repeated tasks
                            tasks = TransTask.objects.filter(
                                content_type=ct,
                                object_class=ob_class_name,
                                object_pk=item.pk,
                                object_field=field,
                                language=language,
                                user=user
                            )
                            for i, task in enumerate(tasks):
                                if i == 0:
                                    task.date_modification = datetime.now()
                                    task.object_field_value = object_field_value
                                    task.done = False
                                    task.save()
                                else:
                                    task.delete()

        # we return every task (created or modified)
        return result