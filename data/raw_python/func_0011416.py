def _pfp__handle_implicit_array(self, name, child):
        """Handle inserting implicit array elements
        """
        existing_child = self._pfp__children_map[name]
        if isinstance(existing_child, Array):
            # I don't think we should check this
            #
            #if existing_child.field_cls != child.__class__:
            #    raise errors.PfpError("implicit arrays must be sequential!")
            existing_child.append(child)
            return existing_child
        else:
            cls = child._pfp__class if hasattr(child, "_pfp__class") else child.__class__
            ary = Array(0, cls)
            # since the array starts with the first item
            ary._pfp__offset = existing_child._pfp__offset
            ary._pfp__parent = self
            ary._pfp__name = name
            ary.implicit = True
            ary.append(existing_child)
            ary.append(child)

            exist_idx = -1
            for idx,child in enumerate(self._pfp__children):
                if child is existing_child:
                    exist_idx = idx
                    break

            self._pfp__children[exist_idx] = ary
            self._pfp__children_map[name] = ary
            return ary