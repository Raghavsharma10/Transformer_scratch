def make_objects(self, meta, data, related_fields=None):
        '''Generator of :class:`stdnet.odm.StdModel` instances with data
from database.

:parameter meta: instance of model :class:`stdnet.odm.Metaclass`.
:parameter data: iterator over instances data.
'''
        make_object = meta.make_object
        related_data = []
        if related_fields:
            for fname, fdata in iteritems(related_fields):
                field = meta.dfields[fname]
                if field in meta.multifields:
                    related = dict(fdata)
                    multi = True
                else:
                    multi = False
                    relmodel = field.relmodel
                    related = dict(((obj.id, obj) for obj in
                                    self.make_objects(relmodel._meta, fdata)))
                related_data.append((field, related, multi))
        for state in data:
            instance = make_object(state, self)
            for field, rdata, multi in related_data:
                if multi:
                    field.set_cache(instance, rdata.get(str(instance.id)))
                else:
                    rid = getattr(instance, field.attname, None)
                    if rid is not None:
                        value = rdata.get(rid)
                        setattr(instance, field.name, value)
            yield instance