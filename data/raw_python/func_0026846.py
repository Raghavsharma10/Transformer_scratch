def _called_thru_default_qs(self, node):
        """Checks if an attribute is being accessed throught the default
        queryset manager, ie: MyClass.objects.filter(some='value')"""
        last_child = node.last_child()
        if not last_child:
            return False

        # the default qs manager is called 'objects', we check for it here
        attrname = getattr(last_child, 'attrname', None)
        if attrname != 'objects':
            return False

        base_cls = last_child.last_child()
        base_classes = DOCUMENT_BASES
        for cls in base_cls.inferred():
            if node_is_subclass(cls, *base_classes):
                return True

        return False