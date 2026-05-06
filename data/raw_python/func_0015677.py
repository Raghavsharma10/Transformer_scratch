def typeinfo_to_ctypes(info, return_value=False):
    """Maps a GITypeInfo() to a ctypes type.

    The ctypes types have to be different in the case of return values
    since ctypes does 'auto unboxing' in some cases which gives
    us no chance to free memory if there is a ownership transfer.
    """

    tag = info.tag.value
    ptr = info.is_pointer

    mapping = {
        GITypeTag.BOOLEAN: gboolean,
        GITypeTag.INT8: gint8,
        GITypeTag.UINT8: guint8,
        GITypeTag.INT16: gint16,
        GITypeTag.UINT16: guint16,
        GITypeTag.INT32: gint32,
        GITypeTag.UINT32: guint32,
        GITypeTag.INT64: gint64,
        GITypeTag.UINT64: guint64,
        GITypeTag.FLOAT: gfloat,
        GITypeTag.DOUBLE: gdouble,
        GITypeTag.VOID: None,
        GITypeTag.GTYPE: GType,
        GITypeTag.UNICHAR: gunichar,
    }

    if ptr:
        if tag == GITypeTag.INTERFACE:
            return gpointer
        elif tag in (GITypeTag.UTF8, GITypeTag.FILENAME):
            if return_value:
                # ctypes does auto conversion to str and gives us no chance
                # to free the pointer if transfer=everything
                return gpointer
            else:
                return gchar_p
        elif tag == GITypeTag.ARRAY:
            return gpointer
        elif tag == GITypeTag.ERROR:
            return GErrorPtr
        elif tag == GITypeTag.GLIST:
            return GListPtr
        elif tag == GITypeTag.GSLIST:
            return GSListPtr
        else:
            if tag in mapping:
                return ctypes.POINTER(mapping[tag])
    else:
        if tag == GITypeTag.INTERFACE:
            iface = info.get_interface()
            iface_type = iface.type.value
            if iface_type == GIInfoType.ENUM:
                return guint32
            elif iface_type == GIInfoType.OBJECT:
                return gpointer
            elif iface_type == GIInfoType.STRUCT:
                return gpointer
            elif iface_type == GIInfoType.UNION:
                return gpointer
            elif iface_type == GIInfoType.FLAGS:
                return guint
            elif iface_type == GIInfoType.CALLBACK:
                return GCallback

            raise NotImplementedError(
                "Could not convert interface: %r to ctypes type" % iface.type)
        else:
            if tag in mapping:
                return mapping[tag]

    raise NotImplementedError("Could not convert %r to ctypes type" % info.tag)