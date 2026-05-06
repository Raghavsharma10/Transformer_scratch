def fragment(self, fragment_size, mask=False):
        """
        Fragment the frame into a chain of fragment frames:
        - An initial frame with non-zero opcode
        - Zero or more frames with opcode = 0 and final = False
        - A final frame with opcode = 0 and final = True

        The first and last frame may be the same frame, having a non-zero
        opcode and final = True. Thus, this function returns a list containing
        at least a single frame.

        `fragment_size` indicates the maximum payload size of each fragment.
        The payload of the original frame is split into one or more parts, and
        each part is converted to a Frame instance.

        `mask` is a boolean (default False) indicating whether the payloads
        should be masked. If True, each frame is assigned a randomly generated
        masking key.
        """
        frames = []

        for start in xrange(0, len(self.payload), fragment_size):
            payload = self.payload[start:start + fragment_size]
            frames.append(Frame(OPCODE_CONTINUATION, payload, mask=mask,
                                final=False))

        frames[0].opcode = self.opcode
        frames[-1].final = True

        return frames