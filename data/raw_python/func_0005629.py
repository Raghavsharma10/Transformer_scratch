def _(mcs, cls_name="Object", with_meta=None):
        """ Method to generate real metaclass to be used::

                mc = ExtensibleType._("MyClass")  # note this line
                @six.add_metaclass(mc)
                class MyClassBase(object):
                    pass

            :param str cls_name: name of generated class
            :param class with_meta: Mix aditional metaclass in.
                                    (default: None)
            :return: specific metaclass to track new inheritance tree
        """
        if with_meta is not None:
            class EXType(with_meta, mcs):
                _cls_name = cls_name
                _base_classes = []
                _generated_class = None
        else:
            class EXType(mcs):
                _cls_name = cls_name
                _base_classes = []
                _generated_class = None

        return EXType