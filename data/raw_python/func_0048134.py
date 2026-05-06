def save_relationship(self, relationship_form, *args, **kwargs):
        """Pass through to provider RelationshipAdminSession.update_relationship"""
        # Implemented from kitosid template for -
        # osid.resource.ResourceAdminSession.update_resource
        if relationship_form.is_for_update():
            return self.update_relationship(relationship_form, *args, **kwargs)
        else:
            return self.create_relationship(relationship_form, *args, **kwargs)