def get_class(mcs):
        """ Generates new class to gether logic of all available extensions
            ::

                mc = ExtensibleType._("MyClass")
                @six.add_metaclass(mc)
                class MyClassBase(object):
                    pass

                # get class with all extensions enabled
                MyClass = mc.get_class()

        """
        if mcs._generated_class is None:
            mcs._generated_class = type(
                mcs._cls_name,
                tuple(mcs._base_classes),
                {'_generated': True})
        return mcs._generated_class