def _pfp__notify_update(self, child=None):
        """Handle a child with an updated value
        """
        if getattr(self, "_pfp__union_update_other_children", True):
            self._pfp__union_update_other_children = False

            new_data = child._pfp__build()
            new_stream =  bitwrap.BitwrappedStream(six.BytesIO(new_data))
            for other_child in self._pfp__children:
                if other_child is child:
                    continue

                if isinstance(other_child, Array) and other_child.is_stringable():
                    other_child._pfp__set_value(new_data)
                else:
                    other_child._pfp__parse(new_stream)
                new_stream.seek(0)

            self._pfp__no_update_other_children = True
        
        super(Union, self)._pfp__notify_update(child=child)