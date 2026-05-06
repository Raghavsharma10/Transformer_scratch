def factory(cls, str_type, version):
        """Return a proper object
        """
        type = Object.Types(str_type)

        if type is Object.Types.object:
            object = ObjectObject()
        elif type is Object.Types.array:
            object = ObjectArray()
        elif type is Object.Types.number:
            object = ObjectNumber()
        elif type is Object.Types.integer:
            object = ObjectInteger()
        elif type is Object.Types.string:
            object = ObjectString()
        elif type is Object.Types.boolean:
            object = ObjectBoolean()
        elif type is Object.Types.reference:
            object = ObjectReference()
        elif type is Object.Types.type:
            object = ObjectType()
        elif type is Object.Types.none:
            object = ObjectNone()
        elif type is Object.Types.dynamic:
            object = ObjectDynamic()
        elif type is Object.Types.const:
            object = ObjectConst()
        elif type is Object.Types.enum:
            object = ObjectEnum()
        else:
            object = Object()
        object.type = type
        object.version = version
        return object