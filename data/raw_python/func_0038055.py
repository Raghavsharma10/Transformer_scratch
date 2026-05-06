def to_internal_value(self, value):
        """Basically, each tag dict must include a full dict with id,
        name and slug--or else you need to pass in a dict with just a name,
        which indicated that the Tag doesn't exist, and should be added."""

        if "id" in value:
            tag = Tag.objects.get(id=value["id"])
        else:
            if "name" not in value:
                raise ValidationError("Tags must include an ID or a name.")
            if len(slugify(value["name"])) > 50:
                raise ValidationError("Maximum tag length is 50 characters.")

            name = value["name"]
            slug = value.get("slug", slugify(name))
            try:
                tag = Tag.objects.get(slug=slug)
            except Tag.DoesNotExist:
                tag, created = Tag.objects.get_or_create(name=name, slug=slug)
        return tag