def create(self, obj, ref=None):
        """ Convert *obj* to a new ShaderObject. If the output is a Variable
        with no name, then set its name using *ref*. 
        """
        if isinstance(ref, Variable):
            ref = ref.name
        elif isinstance(ref, string_types) and ref.startswith('gl_'):
            # gl_ names not allowed for variables
            ref = ref[3:].lower()
        
        # Allow any type of object to be converted to ShaderObject if it
        # provides a magic method:
        if hasattr(obj, '_shader_object'):
            obj = obj._shader_object()
        
        if isinstance(obj, ShaderObject):
            if isinstance(obj, Variable) and obj.name is None:
                obj.name = ref
        elif isinstance(obj, string_types):
            obj = TextExpression(obj)
        else:
            obj = Variable(ref, obj)
            # Try prepending the name to indicate attribute, uniform, varying
            if obj.vtype and obj.vtype[0] in 'auv':
                obj.name = obj.vtype[0] + '_' + obj.name 
        
        return obj