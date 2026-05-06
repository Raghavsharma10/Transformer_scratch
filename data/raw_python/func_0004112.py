def factory(cls, object_raw):
        """Return a proper object
        """
        if object_raw is None:
            return None
        if object_raw.type is ObjectRaw.Types.object:
            return ObjectObject(object_raw)
        elif object_raw.type is ObjectRaw.Types.type:
            return ObjectType(object_raw)
        elif object_raw.type is ObjectRaw.Types.array:
            return ObjectArray(object_raw)
        elif object_raw.type is ObjectRaw.Types.dynamic:
            return ObjectDynamic(object_raw)
        elif object_raw.type is ObjectRaw.Types.const:
            return ObjectConst(object_raw)
        elif object_raw.type is ObjectRaw.Types.enum:
            return ObjectEnum(object_raw)
        else:
            return Object(object_raw)