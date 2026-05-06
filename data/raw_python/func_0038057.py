def to_internal_value(self, data):
        """Basically, each author dict must include either a username or id."""
        # model = get_user_model()
        model = self.Meta.model

        if "id" in data:
            author = model.objects.get(id=data["id"])
        else:
            if "username" not in data:
                raise ValidationError("Authors must include an ID or a username.")
            username = data["username"]
            author = model.objects.get(username=username)
        return author