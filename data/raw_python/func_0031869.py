def create(self, fields):
        """
        Create the object only once.
        So, you need loop to usage.

        :param `fields` is dictionary fields.
        """
        try:
            # Cleaning the fields, and check if has `ForeignKey` type.
            cleaned_fields = {}
            for key, value in fields.items():
                if type(value) is dict:
                    try:
                        if value['type'] == 'fk':
                            fake_fk = self.fake_fk(value['field_name'])
                            cleaned_fields.update({key: fake_fk})
                    except:
                        pass
                else:
                    cleaned_fields.update({key: value})

            # Creating the object from dictionary fields.
            model_class = self.model_class()
            obj = model_class.objects.create(**cleaned_fields)

            # The `ManyToManyField` need specific object,
            # so i handle it after created the object.
            for key, value in fields.items():
                if type(value) is dict:
                    try:
                        if value['type'] == 'm2m':
                            self.fake_m2m(obj, value['field_name'])
                    except:
                        pass
            try:
                obj.save_m2m()
            except:
                obj.save()
            return obj
        except Exception as e:
            raise e