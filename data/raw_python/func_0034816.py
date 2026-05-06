def recv(self):
        """
        Receive a message. A message may consist of multiple (ordered) data
        frames. A control frame may be delivered at any time, also when
        expecting the next continuation frame of a fragmented message. These
        control frames are handled immediately by handle_control_frame().
        """
        fragments = []

        while not len(fragments) or not fragments[-1].final:
            frame = self.sock.recv()

            if isinstance(frame, ControlFrame):
                self.handle_control_frame(frame)
            elif len(fragments) > 0 and frame.opcode != OPCODE_CONTINUATION:
                raise ValueError('expected continuation/control frame, got %s '
                                 'instead' % frame)
            else:
                fragments.append(frame)

        return self.concat_fragments(fragments)