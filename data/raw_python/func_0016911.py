def _trace(self, frame, event, arg):
        """
        Build a record of called functions using the trace mechanism
        """

        # Return if this is not a function call
        if event != 'call':
            return

        # Filter calling and called functions by module names
        src_mod = current_module_name(frame.f_back)
        dst_mod = current_module_name(frame)

        # Avoid tracing the tracer (specifically, call from
        # ContextCallTracer.__exit__ to CallTracer.stop)
        if src_mod == __modulename__ or dst_mod == __modulename__:
            return

        # Apply source and destination module filters
        if not self.srcmodflt.match(src_mod):
            return
        if not self.dstmodflt.match(dst_mod):
            return

        # Get calling and called functions
        src_func = current_function(frame.f_back)
        dst_func = current_function(frame)

        # Filter calling and called functions by qnames
        if not self.srcqnmflt.match(function_qname(src_func)):
            return
        if not self.dstqnmflt.match(function_qname(dst_func)):
            return

        # Get calling and called function full names
        src_name = function_fqname(src_func)
        dst_name = function_fqname(dst_func)

        # Modify full function names if necessary
        if self.fnmsub is not None:
            src_name = re.sub(self.fnmsub[0], self.fnmsub[1], src_name)
            dst_name = re.sub(self.fnmsub[0], self.fnmsub[1], dst_name)

        # Update calling function count
        if src_func is not None:
            if src_name in self.fncts:
                self.fncts[src_name][0] += 1
            else:
                self.fncts[src_name] = [1, 0]

        # Update called function count
        if dst_func is not None and src_func is not None:
            if dst_name in self.fncts:
                self.fncts[dst_name][1] += 1
            else:
                self.fncts[dst_name] = [0, 1]

        # Update caller/calling pair count
        if dst_func is not None and src_func is not None:
            key = (src_name, dst_name)
            if key in self.calls:
                self.calls[key] += 1
            else:
                self.calls[key] = 1