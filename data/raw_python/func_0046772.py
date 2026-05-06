def get_parts(self, parts=None, reference_level=0):
        """Recursively returns a depth-first list of all known magic parts"""
        if parts is None:
            parts = list()
            new_reference_level = reference_level
        else:
            self._level_in_section = self._level + reference_level
            new_reference_level = self._level_in_section
            parts.append(self.my_osid_object)
        if self._child_parts is None:
            if self.has_magic_children():
                self.generate_children()
            else:
                return parts
        for part in self._child_parts:
            part.get_parts(parts, new_reference_level)
            # Don't need to append here, because parts is passed by reference
            # so appending is redundant
            # child_parts = part.get_parts(parts, new_reference_level)
            # known_part_ids = [str(part.ident) for part in parts]
            #
            # for child_part in child_parts:
            #     if str(child_part.ident) not in known_part_ids:
            #         parts.append(child_part)
            #         known_part_ids.append(str(child_part.ident))
        return parts