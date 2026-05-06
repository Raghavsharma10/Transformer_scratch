def normalize(text, variant=VARIANT1, case_sensitive=False):
    """Create a normalized version of `text`.

    With `variant` set to ``VARIANT1`` (default), german umlauts are
    transformed to plain chars: ``ä`` -> ``a``, ``ö`` -> ``o``, ...::

      >>> print(normalize("mäßig"))
      massig

    With `variant` set to ``VARIANT2``, german umlauts are transformed
    ``ä`` -> ``ae``, etc.::

      >>> print(normalize("mäßig", variant=VARIANT2))
      maessig

    All words are turned to lower-case.::

      >>> print(normalize("Maße"))
      masse

    except if `case_sensitive` is set to `True`::

      >>> print(normalize("Maße", case_sensitive=True))
      Masse

    Other chars with diacritics will be returned with the diacritics
    stripped off::

      >>> print(normalize("Česká"))
      ceska


    """
    text = text.replace("ß", "ss")
    if not case_sensitive:
        text = text.lower()
    if variant == VARIANT2:
        for char, repl in (
                ('ä', 'ae'), ('ö', 'oe'), ('ü', 'ue'),
                ('Ä', 'AE'), ('Ö', 'OE'), ('Ü', 'UE')):
            text = text.replace(char, repl)
    text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore")
    return text.decode()