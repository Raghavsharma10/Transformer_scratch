def get_genus_type(self):
        """Overrides get_genus_type of extended object"""
        enclosed_object_id = self.get_enclosed_object_id()
        package = enclosed_object_id.get_identifier_namespace().split('.')[0]
        obj = enclosed_object_id.get_identifier_namespace().split('.')[1]
        return Type(
            authority='OSID.ORG',
            namespace=package,
            identifier=obj,
            display_name=obj,
            display_label=obj,
            description=package + ' ' + obj + ' type',
            domain=package + '.' + obj)