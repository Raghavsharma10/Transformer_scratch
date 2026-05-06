def parse_string_unsafe(s, system=SI):
    """Attempt to parse a string with ambiguous units and try to make a
bitmath object out of it.

This may produce inaccurate results if parsing shell output. For
example `ls` may say a 2730 Byte file is '2.7K'. 2730 Bytes == 2.73 kB
~= 2.666 KiB. See the documentation for all of the important details.

Note the following caveats:

* All inputs are assumed to be byte-based (as opposed to bit based)

* Numerical inputs (those without any units) are assumed to be a
  number of bytes

* Inputs with single letter units (k, M, G, etc) are assumed to be SI
  units (base-10). Set the `system` parameter to `bitmath.NIST` to
  change this behavior.

* Inputs with an `i` character following the leading letter (Ki, Mi,
  Gi) are assumed to be NIST units (base 2)

* Capitalization does not matter

    """
    if not isinstance(s, (str, unicode)) and \
       not isinstance(s, numbers.Number):
        raise ValueError("parse_string_unsafe only accepts string/number inputs but a %s was given" %
                         type(s))

    ######################################################################
    # Is the input simple to parse? Just a number, or a number
    # masquerading as a string perhaps?

    # Test case: raw number input (easy!)
    if isinstance(s, numbers.Number):
        # It's just a number. Assume bytes
        return Byte(s)

    # Test case: a number pretending to be a string
    if isinstance(s, (str, unicode)):
        try:
            # Can we turn it directly into a number?
            return Byte(float(s))
        except ValueError:
            # Nope, this is not a plain number
            pass

    ######################################################################
    # At this point:
    # - the input is also not just a number wrapped in a string
    # - nor is is just a plain number type
    #
    # We need to do some more digging around now to figure out exactly
    # what we were given and possibly normalize the input into a
    # format we can recognize.

    # First we'll separate the number and the unit.
    #
    # Get the index of the first alphabetic character
    try:
        index = list([i.isalpha() for i in s]).index(True)
    except ValueError:  # pragma: no cover
        # If there's no alphabetic characters we won't be able to .index(True)
        raise ValueError("No unit detected, can not parse string '%s' into a bitmath object" % s)

    # Split the string into the value and the unit
    val, unit = s[:index], s[index:]

    # Don't trust anything. We'll make sure the correct 'b' is in place.
    unit = unit.rstrip('Bb')
    unit += 'B'

    # At this point we can expect `unit` to be either:
    #
    # - 2 Characters (for SI, ex: kB or GB)
    # - 3 Caracters (so NIST, ex: KiB, or GiB)
    #
    # A unit with any other number of chars is not a valid unit

    # SI
    if len(unit) == 2:
        # Has NIST parsing been requested?
        if system == NIST:
            # NIST units requested. Ensure the unit begins with a
            # capital letter and is followed by an 'i' character.
            unit = capitalize_first(unit)
            # Insert an 'i' char after the first letter
            _unit = list(unit)
            _unit.insert(1, 'i')
            # Collapse the list back into a 3 letter string
            unit = ''.join(_unit)
            unit_class = globals()[unit]
        else:
            # Default parsing (SI format)
            #
            # Edge-case checking: SI 'thousand' is a lower-case K
            if unit.startswith('K'):
                unit = unit.replace('K', 'k')
            elif not unit.startswith('k'):
                # Otherwise, ensure the first char is capitalized
                unit = capitalize_first(unit)

            # This is an SI-type unit
            if unit[0] in SI_PREFIXES:
                unit_class = globals()[unit]
    # NIST
    elif len(unit) == 3:
        unit = capitalize_first(unit)

        # This is a NIST-type unit
        if unit[:2] in NIST_PREFIXES:
            unit_class = globals()[unit]
    else:
        # This is not a unit we recognize
        raise ValueError("The unit %s is not a valid bitmath unit" % unit)

    try:
        unit_class
    except UnboundLocalError:
        raise ValueError("The unit %s is not a valid bitmath unit" % unit)

    return unit_class(float(val))