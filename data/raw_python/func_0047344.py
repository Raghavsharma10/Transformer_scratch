def _assign_as_root(self, id_):
        """Assign an id_ a root object in the hierarchy"""
        rfc = self._ras.get_relationship_form_for_create(self._phantom_root_id, id_, [])
        rfc.set_display_name('Implicit Root to ' + str(id_) + ' Parent-Child Relationship')
        rfc.set_description(self._relationship_type.get_display_name().get_text() + ' relationship for implicit root and child: ' + str(id_))
        rfc.set_genus_type(self._relationship_type)
        self._ras.create_relationship(rfc)