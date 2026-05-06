def best_prefix(bytes, system=NIST):
    """Return a bitmath instance representing the best human-readable
representation of the number of bytes given by ``bytes``. In addition
to a numeric type, the ``bytes`` parameter may also be a bitmath type.

Optionally select a preferred unit system by specifying the ``system``
keyword. Choices for ``system`` are ``bitmath.NIST`` (default) and
``bitmath.SI``.

Basically a shortcut for:

   >>> import bitmath
   >>> b = bitmath.Byte(12345)
   >>> best = b.best_prefix()

Or:

   >>> import bitmath
   >>> best = (bitmath.KiB(12345) * 4201).best_prefix()
    """
    if isinstance(bytes, Bitmath):
        value = bytes.bytes
    else:
        value = bytes
    return Byte(value).best_prefix(system=system)