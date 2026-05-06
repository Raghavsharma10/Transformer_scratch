def annotate(self, word):
        '''Annotate 'word' for syllabification, stress, weights, and vowels.'''
        info = []  # e.g., [ ('\'nak.su.`tus.ta', 'PUSU', 'HLHL', 'AUUA'), ]

        for syllabification, _ in syllabify(self.normalize(word), stress=True):
            stresses = ''
            weights = ''
            vowels = ''

            for syll in syllable_split(syllabification):

                try:
                    vowels += get_vowel(syll)
                    weights += get_weight(syll)
                    stresses += {'\'': 'P', '`': 'S'}.get(syll[0], 'U')

                except AttributeError:

                    # if the syllable is vowel-less...
                    if syll[-1].isalpha():
                        stresses += '*'
                        weights += '*'
                        vowels += '*'

                    else:
                        stresses += ' '
                        weights += ' '
                        vowels += ' '

            info.append((
                syllabification,
                stresses,
                weights,
                vowels,
                ))

        return info