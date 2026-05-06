def from_other(cls, item):
        """Factory function to return instances of `item` converted into a new
instance of ``cls``. Because this is a class method, it may be called
from any bitmath class object without the need to explicitly
instantiate the class ahead of time.

*Implicit Parameter:*

* ``cls`` A bitmath class, implicitly set to the class of the
  instance object it is called on

*User Supplied Parameter:*

* ``item`` A :class:`bitmath.Bitmath` subclass instance

*Example:*

   >>> import bitmath
   >>> kib = bitmath.KiB.from_other(bitmath.MiB(1))
   >>> print kib
   KiB(1024.0)

        """
        if isinstance(item, Bitmath):
            return cls(bits=item.bits)
        else:
            raise ValueError("The provided items must be a valid bitmath class: %s" %
                             str(item.__class__))