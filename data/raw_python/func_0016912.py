def stop(self):
        """Stop tracing"""

        # Stop tracing
        sys.settrace(None)

        # Build group structure if group filter is defined
        if self.grpflt is not None:
            # Iterate over graph nodes (functions)
            for k in self.fncts:
                # Construct group identity string
                m = self.grpflt.search(k)
                # If group identity string found, append current node
                # to that group
                if m is not None:
                    ms = m.group(0)
                    if ms in self.group:
                        self.group[ms].append(k)
                    else:
                        self.group[ms] = [k, ]