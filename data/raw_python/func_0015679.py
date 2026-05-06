def new_from_memory(cls, data):
        """Takes bytes and returns a GITypelib, or raises GIError"""

        size = len(data)
        copy = g_memdup(data, size)
        ptr = cast(copy, POINTER(guint8))
        try:
            with gerror(GIError) as error:
                return GITypelib._new_from_memory(ptr, size, error)
        except GIError:
            free(copy)
            raise