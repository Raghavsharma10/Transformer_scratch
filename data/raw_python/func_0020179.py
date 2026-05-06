def register_with_model(self, name, model):
        '''Called during the creation of a the :class:`StdModel`
class when :class:`Metaclass` is initialised. It fills
:attr:`Field.name` and :attr:`Field.model`. This is an internal
function users should never call.'''
        if self.name:
            raise FieldError('Field %s is already registered\
 with a model' % self)
        self.name = name
        self.attname = self.get_attname()
        self.model = model
        meta = model._meta
        self.meta = meta
        meta.dfields[name] = self
        meta.fields.append(self)
        if not self.primary_key:
            self.add_to_fields()
        else:
            model._meta.pk = self