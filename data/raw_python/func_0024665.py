def query_device_capacity(device_fd):
    """Create bitmath instances of the capacity of a system block device

Make one or more ioctl request to query the capacity of a block
device. Perform any processing required to compute the final capacity
value. Return the device capacity in bytes as a :class:`bitmath.Byte`
instance.

Thanks to the following resources for help figuring this out Linux/Mac
ioctl's for querying block device sizes:

* http://stackoverflow.com/a/12925285/263969
* http://stackoverflow.com/a/9764508/263969

   :param file device_fd: A ``file`` object of the device to query the
   capacity of (as in ``get_device_capacity(open("/dev/sda"))``).

   :return: a bitmath :class:`bitmath.Byte` instance equivalent to the
   capacity of the target device in bytes.
"""
    if os_name() != 'posix':
        raise NotImplementedError("'bitmath.query_device_capacity' is not supported on this platform: %s" % os_name())

    s = os.stat(device_fd.name).st_mode
    if not stat.S_ISBLK(s):
        raise ValueError("The file descriptor provided is not of a device type")

    # The keys of the ``ioctl_map`` dictionary correlate to possible
    # values from the ``platform.system`` function.
    ioctl_map = {
        # ioctls for the "Linux" platform
        "Linux": {
            "request_params": [
                # A list of parameters to calculate the block size.
                #
                # ( PARAM_NAME , FORMAT_CHAR , REQUEST_CODE )
                ("BLKGETSIZE64", "L", 0x80081272)
                # Per <linux/fs.h>, the BLKGETSIZE64 request returns a
                # 'u64' sized value. This is an unsigned 64 bit
                # integer C type. This means to correctly "buffer" the
                # result we need 64 bits, or 8 bytes, of memory.
                #
                # The struct module documentation include a reference
                # chart relating formatting characters to native C
                # Types. In this case, using the "native size", the
                # table tells us:
                #
                # * Character 'L' - Unsigned Long C Type (u64) - Loads into a Python int type
                #
                # Confirm this character is right by running (on Linux):
                #
                #    >>> import struct
                #    >>> print 8 == struct.calcsize('L')
                #
                # The result should be true as long as your kernel
                # headers define BLKGETSIZE64 as a u64 type (please
                # file a bug report at
                # https://github.com/tbielawa/bitmath/issues/new if
                # this does *not* work for you)
            ],
            # func is how the final result is decided. Because the
            # Linux BLKGETSIZE64 call returns the block device
            # capacity in bytes as an integer value, no extra
            # calculations are required. Simply return the value of
            # BLKGETSIZE64.
            "func": lambda x: x["BLKGETSIZE64"]
        },
        # ioctls for the "Darwin" (Mac OS X) platform
        "Darwin": {
            "request_params": [
                # A list of parameters to calculate the block size.
                #
                # ( PARAM_NAME , FORMAT_CHAR , REQUEST_CODE )
                ("DKIOCGETBLOCKCOUNT", "L", 0x40086419),
                # Per <sys/disk.h>: get media's block count - uint64_t
                #
                # As in the BLKGETSIZE64 example, an unsigned 64 bit
                # integer will use the 'L' formatting character
                ("DKIOCGETBLOCKSIZE", "I", 0x40046418)
                # Per <sys/disk.h>: get media's block size - uint32_t
                #
                # This request returns an unsigned 32 bit integer, or
                # in other words: just a normal integer (or 'int' c
                # type). That should require 4 bytes of space for
                # buffering. According to the struct modules
                # 'Formatting Characters' chart:
                #
                # * Character 'I' - Unsigned Int C Type (uint32_t) - Loads into a Python int type
            ],
            # OS X doesn't have a direct equivalent to the Linux
            # BLKGETSIZE64 request. Instead, we must request how many
            # blocks (or "sectors") are on the disk, and the size (in
            # bytes) of each block. Finally, multiply the two together
            # to obtain capacity:
            #
            #                      n Block * y Byte
            # capacity (bytes)  =            -------
            #                                1 Block
            "func": lambda x: x["DKIOCGETBLOCKCOUNT"] * x["DKIOCGETBLOCKSIZE"]
            # This expression simply accepts a dictionary ``x`` as a
            # parameter, and then returns the result of multiplying
            # the two named dictionary items together. In this case,
            # that means multiplying ``DKIOCGETBLOCKCOUNT``, the total
            # number of blocks, by ``DKIOCGETBLOCKSIZE``, the size of
            # each block in bytes.
        }
    }

    platform_params = ioctl_map[platform.system()]
    results = {}

    for req_name, fmt, request_code in platform_params['request_params']:
        # Read the systems native size (in bytes) of this format type.
        buffer_size = struct.calcsize(fmt)
        # Construct a buffer to store the ioctl result in
        buffer = ' ' * buffer_size

        # This code has been ran on only a few test systems. If it's
        # appropriate, maybe in the future we'll add try/except
        # conditions for some possible errors. Really only for cases
        # where it would add value to override the default exception
        # message string.
        buffer = fcntl.ioctl(device_fd.fileno(), request_code, buffer)

        # Unpack the raw result from the ioctl call into a familiar
        # python data type according to the ``fmt`` rules.
        result = struct.unpack(fmt, buffer)[0]
        # Add the new result to our collection
        results[req_name] = result

    return Byte(platform_params['func'](results))