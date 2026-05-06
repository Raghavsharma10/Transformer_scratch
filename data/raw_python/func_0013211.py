def marshall(self, registry):
        """Returns bytes"""
        result = b""

        for i in registry.get_all():
            # Each message needs to be prefixed with a varint with the size of
            # the message (MetrycType)
            # https://github.com/matttproud/golang_protobuf_extensions/blob/master/ext/encode.go
            # http://zombietetris.de/blog/building-your-own-writedelimitedto-for-python-protobuf/
            body = self.marshall_collector(i).SerializeToString()
            msg = encoder._VarintBytes(len(body)) + body
            result += msg

        return result