async def _expect_command(cls, reader, name):
        """
        Expect a command.

        :param reader: The reader to use.
        :returns: The command data.
        """
        size_type = struct.unpack('B', await reader.readexactly(1))[0]

        if size_type == 0x04:
            size = struct.unpack('!B', await reader.readexactly(1))[0]
        elif size_type == 0x06:
            size = struct.unpack('!Q', await reader.readexactly(8))[0]
        else:
            raise ProtocolError(
                "Unexpected size type: %0x" % size_type,
                fatal=True,
            )

        name_size = struct.unpack('B', await reader.readexactly(1))[0]

        if name_size != len(name):
            raise ProtocolError(
                "Unexpected command name size: %s (expecting %s)" % (
                    name_size,
                    len(name),
                ),
                fatal=True,
            )

        c_name = await reader.readexactly(name_size)

        if c_name != name:
            raise ProtocolError(
                "Unexpected command name: %s (expecting %s)" % (c_name, name),
                fatal=True,
            )

        return await reader.readexactly(size - name_size - 1)