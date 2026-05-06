def _get_field_type(self, key, value):
        """
        Helper to create field object based on value type
        """
        if isinstance(value, bool):
            return BooleanField(name=key)
        elif isinstance(value, int):
            return IntegerField(name=key)
        elif isinstance(value, float):
            return FloatField(name=key)
        elif isinstance(value, str):
            return StringField(name=key)
        elif isinstance(value, time):
            return TimeField(name=key)
        elif isinstance(value, datetime):
            return DateTimeField(name=key)
        elif isinstance(value, date):
            return DateField(name=key)
        elif isinstance(value, timedelta):
            return TimedeltaField(name=key)
        elif isinstance(value, Enum):
            return EnumField(name=key, enum_class=type(value))
        elif isinstance(value, (dict, BaseDynamicModel, Mapping)):
            return ModelField(name=key, model_class=self.__dynamic_model__ or self.__class__)
        elif isinstance(value, BaseModel):
            return ModelField(name=key, model_class=value.__class__)
        elif isinstance(value, (list, set, ListModel)):
            if not len(value):
                return None
            field_type = self._get_field_type(None, value[0])
            return ArrayField(name=key, field_type=field_type)
        elif value is None:
            return None
        else:
            raise TypeError("Invalid parameter: %s. Type not supported." % (key,))