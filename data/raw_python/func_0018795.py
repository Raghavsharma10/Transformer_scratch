def isready(self, obj) -> bool:
        """Return |True| or |False| to indicate if the protected
        property is ready for the given object.  If the object is
        unknow, |ProtectedProperty| returns |False|."""
        return vars(obj).get(self.name, False)