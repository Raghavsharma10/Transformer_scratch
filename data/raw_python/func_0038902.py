def save(self, **kwargs):
        """
        Method that creates the translations tasks for every selected instance

        :param kwargs:
        :return:
        """
        try:
            # result_ids = []
            manager = Manager()
            for item in self.model_class.objects.language(manager.get_main_language()).filter(pk__in=self.ids).all():
                create_translations_for_item_and_its_children.delay(self.model_class, item.pk, self.languages,
                                                                    update_item_languages=True)
            # return TransTaskSerializer(TransTask.objects.filter(pk__in=result_ids), many=True).data
            return {'status': 'ok'}
        except Exception as e:
            raise serializers.ValidationError(detail=str(e))