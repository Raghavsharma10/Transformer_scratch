def _do_generate(self, source_list, hang_type, crashed_thread, delimiter=' | '):
        """
        each element of signatureList names a frame in the crash stack; and is:
          - a prefix of a relevant frame: Append this element to the signature
          - a relevant frame: Append this element and stop looking
          - irrelevant: Append this element only after seeing a prefix frame
        The signature is a ' | ' separated string of frame names.
        """
        notes = []
        debug_notes = []

        # shorten source_list to the first signatureSentinel
        sentinel_locations = []
        for a_sentinel in self.signature_sentinels:
            if type(a_sentinel) == tuple:
                a_sentinel, condition_fn = a_sentinel
                if not condition_fn(source_list):
                    continue
            try:
                sentinel_locations.append(source_list.index(a_sentinel))
            except ValueError:
                pass
        if sentinel_locations:
            min_index = min(sentinel_locations)
            debug_notes.append(
                'sentinel; starting at "{}" index {}'.format(source_list[min_index], min_index)
            )
            source_list = source_list[min_index:]

        # Get all the relevant frame signatures. Note that these function signatures
        # have already been normalized at this point.
        new_signature_list = []
        for a_signature in source_list:
            # If the signature matches the irrelevant signatures regex, skip to the next frame.
            if self.irrelevant_signature_re.match(a_signature):
                debug_notes.append('irrelevant; ignoring: "{}"'.format(a_signature))
                continue

            # If the frame signature is a dll, remove the @xxxxx part.
            if '.dll' in a_signature.lower():
                a_signature = a_signature.split('@')[0]

                # If this trimmed DLL signature is the same as the previous frame's, skip it.
                if new_signature_list and a_signature == new_signature_list[-1]:
                    continue

            new_signature_list.append(a_signature)

            # If the signature does not match the prefix signatures regex, then it is the last
            # one we add to the list.
            if not self.prefix_signature_re.match(a_signature):
                debug_notes.append('not a prefix; stop: "{}"'.format(a_signature))
                break

            debug_notes.append('prefix; continue iterating: "{}"'.format(a_signature))

        # Add a special marker for hang crash reports.
        if hang_type:
            debug_notes.append(
                'hang_type {}: prepending {}'.format(hang_type, self.hang_prefixes[hang_type])
            )
            new_signature_list.insert(0, self.hang_prefixes[hang_type])

        signature = delimiter.join(new_signature_list)

        # Handle empty signatures to explain why we failed generating them.
        if signature == '' or signature is None:
            if crashed_thread is None:
                notes.append(
                    "CSignatureTool: No signature could be created because we do not know which "
                    "thread crashed"
                )
                signature = "EMPTY: no crashing thread identified"
            else:
                notes.append(
                    "CSignatureTool: No proper signature could be created because no good data "
                    "for the crashing thread ({}) was found".format(crashed_thread)
                )
                try:
                    signature = source_list[0]
                except IndexError:
                    signature = "EMPTY: no frame data available"

        return signature, notes, debug_notes