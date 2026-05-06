def register(self, name, encoder, decoder, content_type,
                 content_encoding='utf-8'):
        """Register a new encoder/decoder.

        :param name: A convenience name for the serialization method.

        :param encoder: A method that will be passed a python data structure
            and should return a string representing the serialized data.
            If ``None``, then only a decoder will be registered. Encoding
            will not be possible.

        :param decoder: A method that will be passed a string representing
            serialized data and should return a python data structure.
            If ``None``, then only an encoder will be registered.
            Decoding will not be possible.

        :param content_type: The mime-type describing the serialized
            structure.

        :param content_encoding: The content encoding (character set) that
            the :param:`decoder` method will be returning. Will usually be
            ``utf-8``, ``us-ascii``, or ``binary``.

        """
        if encoder:
            self._encoders[name] = (content_type, content_encoding, encoder)
        if decoder:
            self._decoders[content_type] = decoder