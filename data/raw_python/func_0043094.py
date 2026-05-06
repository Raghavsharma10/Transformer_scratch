def is_signature_equal(cls, sig_a, sig_b):
        """Compares two signatures using a constant time algorithm to avoid timing attacks."""
        if len(sig_a) != len(sig_b):
            return False

        invalid_chars = 0
        for char_a, char_b in zip(sig_a, sig_b):
            if char_a != char_b:
                invalid_chars += 1
        return invalid_chars == 0