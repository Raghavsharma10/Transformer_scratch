def header2dict(self, names, struct_format, data):
        """
        Unpack the raw received IP and ICMP header information to a dict.
        """
        unpacked_data = struct.unpack(struct_format, data)
        return dict(zip(names, unpacked_data))