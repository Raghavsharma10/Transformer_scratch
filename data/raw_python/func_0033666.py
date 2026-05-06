def build_packet(packet_type, gateway, bulb, payload_fmt, *payload_args,
                 **kwargs):
    """
    Constructs a Lifx packet, returning a bytestring. The arguments are as
    follows:

    - `packet_type`, an integer
    - `gateway`, a 6-byte string containing the mac address of the gateway bulb
    - `bulb`, a 6-byte string containing either the mac address of the target
      bulb or `ALL_BULBS`
    - `payload_fmt`, a `struct`-compatible string that describes the format
      of the payload part of the packet
    - `payload_args`, the values to use to build the payload part of the packet

    Additionally, the `protocol` keyword argument can be used to override the
    protocol field in the packet.
    """
    protocol = kwargs.get('protocol', COMMAND_PROTOCOL)

    packet_fmt = BASE_FORMAT + payload_fmt
    packet_size = struct.calcsize(packet_fmt)
    return struct.pack(packet_fmt,
                       packet_size,
                       protocol,
                       bulb,
                       gateway,
                       0,  # timestamp
                       packet_type,
                       *payload_args)