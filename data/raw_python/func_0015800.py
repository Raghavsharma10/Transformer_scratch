def add_relationship_methods(self):
        """
        Adds relationship methods to applicable model classes.
        """
        Entry = apps.get_model('wagtailrelations', 'Entry')

        @cached_property
        def related(instance):
            return instance.get_related()

        @cached_property
        def related_live(instance):
            return instance.get_related_live()

        @cached_property
        def related_with_scores(instance):
            return instance.get_related_with_scores()

        def get_related(instance):
             entry = Entry.objects.get_for_model(instance)[0]
             return entry.get_related()

        def get_related_live(instance):
             entry = Entry.objects.get_for_model(instance)[0]
             return entry.get_related_live()

        def get_related_with_scores(instance):
            try:
                entry = Entry.objects.get_for_model(instance)[0]
                return entry.get_related_with_scores()
            except IntegrityError:
                return []

        for model in self.applicable_models:
            model.add_to_class(
                'get_related',
                get_related
            )
            model.add_to_class(
                'get_related_live',
                get_related_live
            )
            model.add_to_class(
                'get_related_with_scores',
                get_related_with_scores
            )
            model.add_to_class(
                'related',
                related
            )
            model.add_to_class(
                'related_live',
                related_live
            )
            model.add_to_class(
                'related_with_scores',
                related_with_scores
            )