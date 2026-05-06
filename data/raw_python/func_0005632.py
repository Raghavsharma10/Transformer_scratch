def _(mcs, cls_name='Object', with_meta=None, hashattr='_name'):
        """ Method to generate real metaclass to be used
            ::

                # Create metaclass *mc*
                mc = ExtensibleByHashType._("MyClass", hashattr='name')

                # Create class using *mc* as metaclass
                @six.add_metaclass(mc)
                class MyClassBase(object):
                    pass

            :param str cls_name: name of generated class
            :param class with_meta: Mix aditional metaclass in.
                                    (default: None)
            :param hashattr: name of class Meta attribute to be used as hash.
                             default='_name'
            :return: specific metaclass to track new inheritance tree
        """
        extype = super(ExtensibleByHashType, mcs)._(cls_name=cls_name,
                                                    with_meta=with_meta)

        class EXHType(extype):
            _hashattr = hashattr
            _base_classes_hash = collections.defaultdict(list)

            # Override it by dict to store diferent
            # base generated class for each hash
            _generated_class = {}

        return EXHType