def split_signature(klass, signature):
        """Return a list of the element signatures of the topmost signature tuple.

        If the signature is not a tuple, it returns one element with the entire
        signature. If the signature is an empty tuple, the result is [].

        This is useful for e. g. iterating over method parameters which are
        passed as a single Variant.
        """
        if signature == '()':
            return []

        if not signature.startswith('('):
            return [signature]

        result = []
        head = ''
        tail = signature[1:-1]  # eat the surrounding ()
        while tail:
            c = tail[0]
            head += c
            tail = tail[1:]

            if c in ('m', 'a'):
                # prefixes, keep collecting
                continue
            if c in ('(', '{'):
                # consume until corresponding )/}
                level = 1
                up = c
                if up == '(':
                    down = ')'
                else:
                    down = '}'
                while level > 0:
                    c = tail[0]
                    head += c
                    tail = tail[1:]
                    if c == up:
                        level += 1
                    elif c == down:
                        level -= 1

            # otherwise we have a simple type
            result.append(head)
            head = ''

        return result