def get_sequence(cls, entry):
        """
        get models.Sequence object from XML node entry

        :param entry: XML node entry
        :return: :class:`pyuniprot.manager.models.Sequence` object
        """
        seq_tag = entry.find("./sequence")
        seq = seq_tag.text
        seq_tag.clear()
        return models.Sequence(sequence=seq)