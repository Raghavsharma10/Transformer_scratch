def update(self, obj, data):
        """Helper function to update an already existing document
    instead of creating a new one.
    :param obj: Mongoengine Document to update
    :param data: incomming payload to deserialize
    :return: an :class UnmarshallResult:

    Example: ::

        from marshmallow_mongoengine import ModelSchema
        from mymodels import User

        class UserSchema(ModelSchema):
            class Meta:
                model = User

        def update_obj(id, payload):
            user = User.objects(id=id).first()
            result = UserSchema().update(user, payload)
            result.data is user # True

    Note:

        Given the update is done on a existing object, the required param
        on the fields is ignored
        """
        # TODO: find a cleaner way to skip required validation on update
        required_fields = [k for k, f in self.fields.items() if f.required]
        for field in required_fields:
            self.fields[field].required = False
        loaded_data, errors = self._do_load(data, postprocess=False)
        for field in required_fields:
            self.fields[field].required = True
        if not errors:
            # Update the given obj fields
            for k, v in loaded_data.items():
                # Skip default values that have been automatically
                # added during unserialization
                if k in data:
                    setattr(obj, k, v)
        return ma.UnmarshalResult(data=obj, errors=errors)