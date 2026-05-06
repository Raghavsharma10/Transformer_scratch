def chop_sequence(sequence, limit_length):
        """Input sequence is divided on smaller non-overlapping sequences with set length.  """
        return [sequence[i:i + limit_length] for i in range(0, len(sequence), limit_length)]