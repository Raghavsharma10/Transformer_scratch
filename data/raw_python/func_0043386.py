def write_command(cls, writer, name, buffers=()):
        """
        Write a command to the specified writer.

        :param writer: The writer to use.
        :param name: The command name.
        :param buffers: The buffers to writer.
        """
        assert len(name) < 256

        body_len = len(name) + 1 + sum(len(buffer) for buffer in buffers)

        if body_len < 256:
            writer.write(struct.pack('!BBB', 0x04, body_len, len(name)))
        else:
            writer.write(struct.pack('!BQB', 0x06, body_len, len(name)))

        writer.write(name)

        for buffer in buffers:
            writer.write(buffer)