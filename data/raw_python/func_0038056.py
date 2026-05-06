def to_internal_value(self, value):
        """Basically, each tag dict must include a full dict with id,
        name and slug--or else you need to pass in a dict with just a name,
        which indicated that the FeatureType doesn't exist, and should be added."""
        if value == "":
            return None

        if isinstance(value, string_types):
            slug = slugify(value)
            feature_type, created = FeatureType.objects.get_or_create(
                slug=slug,
                defaults={"name": value}
            )
        else:
            if "id" in value:
                feature_type = FeatureType.objects.get(id=value["id"])
            elif "slug" in value:
                feature_type = FeatureType.objects.get(slug=value["slug"])
            else:
                raise ValidationError("Invalid feature type data")
        return feature_type