def stress(syllabified_simplex_word):
    '''Assign primary and secondary stress to 'syllabified_simplex_word'.'''
    syllables = syllabified_simplex_word.split('.')
    stressed = '\'' + syllables[0]  # primary stress

    try:
        n = 0
        medial = syllables[1:-1]

        for i, syll in enumerate(medial):

            if (i + n) % 2 == 0:
                stressed += '.' + syll

            else:
                try:
                    if is_light(syll) and is_heavy(medial[i + 1]):
                        stressed += '.' + syll
                        n += 1
                        continue

                except IndexError:
                    pass

                # secondary stress
                stressed += '.`' + syll

    except IndexError:
        pass

    if len(syllables) > 1:
        stressed += '.' + syllables[-1]

    return stressed