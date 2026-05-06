def get_class(mcs, name, default=False):
        """ Generates new class to gether logic of all available extensions
            ::

                # Create metaclass *mc*
                mc = ExtensibleByHashType._("MyClass", hashattr='name')

                # Use metaclass *mc* to create base class for extensions
                @six.add_metaclass(mc)
                class MyClassBase(object):
                    pass

                # Create extension
                class MyClassX1(MyClassBase):
                    class Meta:
                        name = 'X1'

                # get default class
                MyClass = mc.get_class(None, default=True)

                # get specific class
                MyX1 = mc.get_class('X1')

            :param name: key to get class for
            :param bool default: if set to True will generate default class for
                                 if there no special class defined for such key
            :return: generated class for requested type
        """
        if default is False and name not in mcs._base_classes_hash:
            raise ValueError(
                "There is no class registered for key '%s'" % name)
        if mcs._generated_class.get(name, None) is None:
            cls = type(
                mcs._cls_name,
                tuple(mcs._get_base_classes(name)),
                {'_generated': True})
            mcs._generated_class[name] = cls
        return mcs._generated_class[name]