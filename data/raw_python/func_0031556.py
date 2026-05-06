def randomBinaryField(self):
        """
        Return random bytes format.
        """
        lst = [
            b"hello world",
            b"this is bytes",
            b"awesome django",
            b"djipsum is awesome",
            b"\x00\x01\x02\x03\x04\x05\x06\x07",
            b"\x0b\x0c\x0e\x0f"
        ]
        return self.randomize(lst)