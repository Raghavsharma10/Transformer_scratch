def _build_superclass_lists(self):
        """
        >>> CodeBaseDoc(['examples']).all_classes['MySubClass'].all_superclasses[0].name
        'MyClass'
        """
        cls_dict = self.all_classes
        for cls in list(cls_dict.values()):
            cls.all_superclasses = []
            superclass = cls.superclass
            try:
                while superclass:
                    superclass_obj = cls_dict[superclass]
                    cls.all_superclasses.append(superclass_obj)
                    superclass = superclass_obj.superclass
            except KeyError:
                print("Missing superclass: " + superclass)