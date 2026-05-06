def fromType(datatype):
        """Use this method to create a MultiValue type from a specific GP Value
           type.
           
              >>> gpmultistr = arcrest.GPMultiValue.fromType(arcrest.GPString)
              >>> gpvalue = gpmultistr(["a", "b", "c"])
        """
        if issubclass(datatype, GPBaseType):
            return GPBaseType._get_type_by_name("GPMultiValue:%s" % 
                                                datatype.__name__)
        else:
            return GPBaseType._get_type_by_name("GPMultiValue:%s" % 
                                                str(datatype))