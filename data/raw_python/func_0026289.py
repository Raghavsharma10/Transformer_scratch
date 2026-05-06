def get_all_keys(self, start=None):
        """
        A generator which yields a list of all valid keys starting at the
        given `start` offset.  If `start` is `None`, we will start from
        the root of the tree.
        """
        s = self.stream
        if not start:
            start = HEADER_SIZE + self.block_size * self.root_block
        s.seek(start)
        block_type = s.read(2)
        if block_type == LEAF:
            reader = LeafReader(self)
            num_keys = struct.unpack('>i', reader.read(4))[0]
            for _ in range(num_keys):
                cur_key = reader.read(self.key_size)
                # We to a tell/seek here so that the user can read from
                # the file while this loop is still being run
                cur_pos = s.tell()
                yield cur_key
                s.seek(cur_pos)
                length = sbon.read_varint(reader)
                reader.seek(length, 1)
        elif block_type == INDEX:
            (_, num_keys, first_child) = struct.unpack('>Bii', s.read(9))
            children = [first_child]
            for _ in range(num_keys):
                # Skip the key field.
                _ = s.read(self.key_size)
                # Read pointer to the child block.
                next_child = struct.unpack('>i', s.read(4))[0]
                children.append(next_child)
            for child_loc in children:
                for key in self.get_all_keys(HEADER_SIZE + self.block_size * child_loc):
                    yield key
        elif block_type == FREE:
            pass
        else:
            raise Exception('Unhandled block type: {}'.format(block_type))