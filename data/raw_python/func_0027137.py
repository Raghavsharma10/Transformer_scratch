def get_fields(self):
        """
        When static declaration is used, event type choices are fetched too early -
        even before all apps are initialized. As a result, some event types are missing.
        When dynamic declaration is used, all valid event types are available as choices.
        """
        fields = super(BaseHookSerializer, self).get_fields()
        fields['event_types'] = serializers.MultipleChoiceField(
            choices=loggers.get_valid_events(), required=False)
        fields['event_groups'] = serializers.MultipleChoiceField(
            choices=loggers.get_event_groups_keys(), required=False)
        return fields