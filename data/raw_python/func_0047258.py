def _unfold_map(self, display_text_map):
        """Parses a display text dictionary map."""
        from ..type.primitives import Type
        lt_identifier = Id(display_text_map['languageTypeId']).get_identifier()
        st_identifier = Id(display_text_map['scriptTypeId']).get_identifier()
        ft_identifier = Id(display_text_map['formatTypeId']).get_identifier()
        try:
            self._language_type = Type(**language_types.get_type_data(lt_identifier))
        except AttributeError:
            raise NotFound('Language Type: ' + lt_identifier)  # or move on to another source
        try:
            self._script_type = Type(**script_types.get_type_data(st_identifier))
        except AttributeError:
            raise NotFound('Script Type: ' + st_identifier)  # or move on to another source
        try:
            self._format_type = Type(**format_types.get_type_data(ft_identifier))
        except AttributeError:
            raise NotFound('Format Type: ' + ft_identifier)  # or move on to another source
        self._text = display_text_map['text']