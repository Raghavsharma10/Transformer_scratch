def get_field_info(self, field):
        """
        This method is basically a mirror from rest_framework==3.3.3

        We are currently pinned to rest_framework==3.1.1. If we upgrade,
        this can be refactored and simplified to rely more heavily on
        rest_framework's built in logic.
        """

        field_info = self.get_attributes(field)
        field_info["required"] = getattr(field, "required", False)
        field_info["type"] = self.get_label_lookup(field)

        if getattr(field, "child", None):
            field_info["child"] = self.get_field_info(field.child)
        elif getattr(field, "fields", None):
            field_info["children"] = self.get_serializer_info(field)

        if (not isinstance(field, (serializers.RelatedField, serializers.ManyRelatedField)) and
                hasattr(field, "choices")):
            field_info["choices"] = [
                {
                    "value": choice_value,
                    "display_name": force_text(choice_name, strings_only=True)
                }
                for choice_value, choice_name in field.choices.items()
            ]

        return field_info