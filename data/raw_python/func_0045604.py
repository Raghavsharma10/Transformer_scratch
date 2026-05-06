def emit_function(self, return_type=None, argtypes=[], proxy=True):
        """Compiles code and returns a Python-callable function."""

        if argtypes is not None:
            make_func = ctypes.CFUNCTYPE(return_type, *argtypes)
        else:
            make_func = ctypes.CFUNCTYPE(return_type)

        # NOTE: An optional way of binding the function is to use cffi.
        # It's a tad faster then emit_function_fast:
        #   import cffi
        #   ffi = cffi.FFI()
        #   ...
        #   code = jit.emit()
        #   func = ffi.cast("long (*fptr)(long, long)", code.value)
        #   func(123)

        code = self.emit()
        func = make_func(code.value)

        # Save this in case anyone wants to disassemble using external
        # libraries
        func.address = code

        # Because functions code are munmapped when we call _jit_destroy_state,
        # we need to return weakrefs to the functions. Otherwise, a user could
        # call a function that points to invalid memory.
        if proxy:
            self.functions.append(func)
            return weakref.proxy(func)
        else:
            return func