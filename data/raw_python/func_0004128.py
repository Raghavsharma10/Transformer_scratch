def factory(cls, object_source):
        """Return a proper object
        """
        if object_source.type is ObjectRaw.Types.object:
            return ObjectObject(object_source)
        elif object_source.type not in ObjectRaw.Types or object_source.type is ObjectRaw.Types.type:
            return ObjectType(object_source)
        elif object_source.type is ObjectRaw.Types.array:
            return ObjectArray(object_source)
        elif object_source.type is ObjectRaw.Types.dynamic:
            return ObjectDynamic(object_source)
        elif object_source.type is ObjectRaw.Types.const:
            return ObjectConst(object_source)
        elif object_source.type is ObjectRaw.Types.enum:
            return ObjectEnum(object_source)
        else:
            return Object(object_source)