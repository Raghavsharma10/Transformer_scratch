def _load_jamo_short_names():
    """
    Function for parsing the Jamo short names from the Unicode Character Database (UCD) and generating a lookup table
    For more info on how this is used, see the Unicode Standard, ch. 03, section 3.12, "Conjoining Jamo Behavior" and
    ch. 04, section 4.8, "Name".

    https://www.unicode.org/versions/latest/ch03.pdf
    https://www.unicode.org/versions/latest/ch04.pdf
    """
    filename = "Jamo.txt"
    current_dir = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(current_dir, filename), mode="r", encoding="utf-8") as fp:
        for line in fp:
            if not line.strip() or line.startswith("#"):
                continue  # Skip empty lines or lines that are comments (comments start with '#')
            data = line.strip().split(";")
            code = int(data[0].strip(), 16)
            char_info = data[1].split("#")
            short_name = char_info[0].strip()
            _jamo_short_names[code] = short_name