def create_for_object(self, parent_object, slot, role='m', title=None):
        """
        Create a placeholder with the given parameters
        """
        from .db import Placeholder
        parent_attrs = get_parent_lookup_kwargs(parent_object)
        obj = self.create(
            slot=slot,
            role=role or Placeholder.MAIN,
            title=title or slot.title().replace('_', ' '),
            **parent_attrs
        )
        obj.parent = parent_object  # fill the reverse cache
        return obj