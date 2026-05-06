def get_translatable_children(self, obj):
        """
        Obtain all the translatable children from "obj"

        :param obj:
        :return:
        """
        collector = NestedObjects(using='default')
        collector.collect([obj])
        object_list = collector.nested()
        items = self.get_elements(object_list)
        # avoid first object because it's the main object
        return items[1:]