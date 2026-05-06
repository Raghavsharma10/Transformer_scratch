def normalize_frame(
        self,
        module=None,
        function=None,
        file=None,
        line=None,
        module_offset=None,
        offset=None,
        normalized=None,
        **kwargs  # eat any extra kwargs passed in
    ):
        """Normalizes a single frame

        Returns a structured conglomeration of the input parameters to serve as
        a signature. The parameter names of this function reflect the exact
        names of the fields from the jsonMDSW frame output. This allows this
        function to be invoked by passing a frame as ``**a_frame``.

        Sometimes, a frame may already have a normalized version cached. If
        that exists, return it instead.

        """
        # If there's a cached normalized value, use that so we don't spend time
        # figuring it out again
        if normalized is not None:
            return normalized

        if function:
            # If there's a filename and it ends in .rs, then normalize using
            # Rust rules
            if file and (parse_source_file(file) or '').endswith('.rs'):
                return self.normalize_rust_function(
                    function=function,
                    line=line
                )

            # Otherwise normalize it with C/C++ rules
            return self.normalize_cpp_function(
                function=function,
                line=line
            )

        # If there's a file and line number, use that
        if file and line:
            filename = file.rstrip('/\\')
            if '\\' in filename:
                file = filename.rsplit('\\')[-1]
            else:
                file = filename.rsplit('/')[-1]
            return '{}#{}'.format(file, line)

        # If there's an offset and no module/module_offset, use that
        if not module and not module_offset and offset:
            return '@{}'.format(offset)

        # Return module/module_offset
        return '{}@{}'.format(module or '', module_offset)