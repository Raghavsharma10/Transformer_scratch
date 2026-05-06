def pop_refs(self, *refs):
        """Decrement the usage of each ref by 1.

        If this was the last use of a ref, remove it from attr_names or attr_values.
        """
        for ref in refs:
            name = ref.name
            count = self.counts[name]
            # Not tracking this ref
            if count < 1:
                continue
            # Someone else is using this ref
            elif count > 1:
                self.counts[name] -= 1
            # Last reference
            else:
                logger.debug("popping last usage of {}".format(ref))
                self.counts[name] -= 1
                if ref.type == "value":
                    del self.attr_values[name]
                else:
                    # Clean up both name indexes
                    path_segment = self.attr_names[name]
                    del self.attr_names[name]
                    del self.name_attr_index[path_segment]