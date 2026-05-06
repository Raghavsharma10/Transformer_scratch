def add_accent(components, accent):
    """
    Add accent to the given components. The parameter components is
    the result of function separate()
    """
    vowel = components[1]
    last_consonant = components[2]
    if accent == Accent.NONE:
        vowel = remove_accent_string(vowel)
        return [components[0], vowel, last_consonant]

    if vowel == "":
        return components
    #raw_string is a list, not a str object
    raw_string = remove_accent_string(vowel).lower()
    new_vowel = ""
    # Highest priority for ê and ơ
    index = max(raw_string.find("ê"), raw_string.find("ơ"))
    if index != -1:
        new_vowel = vowel[:index] + add_accent_char(vowel[index], accent) + vowel[index+1:]
    elif len(vowel) == 1 or (len(vowel) == 2 and last_consonant == ""):
        new_vowel = add_accent_char(vowel[0], accent) + vowel[1:]
    else:
        new_vowel = vowel[:1] + add_accent_char(vowel[1], accent) + vowel[2:]
    return [components[0], new_vowel, components[2]]