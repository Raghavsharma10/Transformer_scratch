def _related(self, concept):
        """
        Returns related concepts for a concept.
        """
        return concept.hypernyms() + \
                concept.hyponyms() + \
                concept.member_meronyms() + \
                concept.substance_meronyms() + \
                concept.part_meronyms() + \
                concept.member_holonyms() + \
                concept.substance_holonyms() + \
                concept.part_holonyms() + \
                concept.attributes() + \
                concept.also_sees() + \
                concept.similar_tos()