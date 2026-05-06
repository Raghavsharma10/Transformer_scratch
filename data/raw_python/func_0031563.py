def create_validated_fields(self):
        """
        To generate lorem ipsum by validated fields for the model.
        """
        model_class = self.model_class
        fields = self.fields
        maximum = self.maximum
        objects = []

        for n in range(maximum):
            data_dict = {}
            for field in fields:

                def default_assign(func):
                    data_dict[field['field_name']] = func

                def string_assign(func):
                    data_dict[field['field_name']] = str(func)

                if field['field_type'] == 'BigIntegerField':  # values from -9223372036854775808 to 9223372036854775807
                    default_assign(random.randint(-9223372036854775808, 9223372036854775807))
                elif field['field_type'] == 'BinaryField':  # b'', self.randomBinaryField()
                    default_assign(self.randomBinaryField())
                elif field['field_type'] == 'BooleanField':  # True/False
                    default_assign(self.randomize([True, False]))
                elif field['field_type'] == 'CharField':  # self.randomCharField()
                    string_assign(self.randomCharField(model_class, field['field_name']))
                elif field['field_type'] == 'CommaSeparatedIntegerField':  # self.randomCommaSeparatedIntegerField()
                    string_assign(self.randomCommaSeparatedIntegerField())
                elif field['field_type'] == 'DateField':  # '2016-10-11'
                    string_assign(str(datetime.datetime.now().date()))
                elif field['field_type'] == 'DateTimeField':  # '2016-10-11 00:44:08.864285'
                    string_assign(str(datetime.datetime.now()))
                elif field['field_type'] == 'DecimalField':  # self.randomDecimalField()
                    default_assign(self.randomDecimalField(model_class, field['field_name']))
                elif field['field_type'] == 'DurationField':  # such as 1 day, 4 days or else.
                    default_assign(datetime.timedelta(days=random.randint(1, 10)))
                elif field['field_type'] == 'EmailField':  # self.randomEmailField()
                    string_assign(self.randomEmailField())
                elif field['field_type'] == 'FileField':  # self.randomFileField()
                    string_assign(self.randomFileField())
                elif field['field_type'] == 'FloatField':  # 1.92, 0.0, 5.0, or else.
                    default_assign(float(("%.2f" % float(random.randint(0, 100) / 13))))
                elif field['field_type'] == 'ImageField':  # self.randomImageField()
                    string_assign(self.randomImageField())
                elif field['field_type'] == 'IntegerField':  # values from -2147483648 to 2147483647
                    default_assign(random.randint(-2147483648, 2147483647))
                elif field['field_type'] == 'GenericIPAddressField':  # self.randomGenericIPAddressField()
                    string_assign(self.randomGenericIPAddressField())
                elif field['field_type'] == 'NullBooleanField':  # by Default is None/null
                    default_assign(self.randomize([None, True, False]))
                elif field['field_type'] == 'PositiveIntegerField':  # values from 0 to 2147483647
                    default_assign(random.randint(0, 2147483647))
                elif field['field_type'] == 'PositiveSmallIntegerField':  # values from 0 to 32767
                    default_assign(random.randint(0, 32767))
                elif field['field_type'] == 'SlugField':  # self.randomSlugField()
                    string_assign(self.randomSlugField())
                elif field['field_type'] == 'SmallIntegerField':  # values from -32768 to 32767
                    default_assign(random.randint(-32768, 32767))
                elif field['field_type'] == 'TextField':  # self.randomTextField()
                    string_assign(self.randomTextField())
                elif field['field_type'] == 'TimeField':  # accepts the same as DateField
                    string_assign(str(datetime.datetime.now().date()))
                elif field['field_type'] == 'URLField':  # self.randomURLField()
                    string_assign(self.randomURLField())
                elif field['field_type'] == 'UUIDField':  # self.randomUUIDField()
                    string_assign(self.randomUUIDField())
                elif field['field_type'] == 'ForeignKey':  # self.getOrCreateForeignKey()
                    default_assign(self.getOrCreateForeignKey(model_class, field['field_name']))
                # elif field['field_type'] == 'OneToOneField':  # pk/id -> not fixed yet.
                #    default_assign(self.randomize([1, ]))
                # Unsolved: need specific pk/id

            obj = model_class.objects.create(**data_dict)

            # Because the Relationship Model need specific id from the object,
            # so, i handle it after created the object.
            for field in fields:
                if field['field_type'] == 'ManyToManyField':
                    # Find the instance model field from `obj` already created before.
                    instance_m2m = getattr(obj, field['field_name'])
                    objects_m2m = instance_m2m.model.objects.all()

                    # Djipsum only create the `ManyToManyField` if the related object is exists.
                    if objects_m2m.exists():
                        ids_m2m = [i.pk for i in objects_m2m]
                        random_decission = random.sample(
                            range(min(ids_m2m), max(ids_m2m)), max(ids_m2m) - 1
                        )
                        # Let me know if the `random_decission` has minimum objects to be choice.
                        if len(random_decission) <= 2:
                            random_decission = [self.randomize(ids_m2m)]
                        related_objects = [
                            rel_obj for rel_obj in objects_m2m
                            if rel_obj.pk in random_decission
                        ]
                        # adding the `ManyToManyField`
                        instance_m2m.add(*related_objects)
            try:
                obj.save_m2m()
            except:
                obj.save()
            objects.append(obj)
        return objects