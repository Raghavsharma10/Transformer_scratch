def find_tokens(string, punctuation=PUNCTUATION, abbreviations=ABBREVIATIONS, replace=replacements, linebreak=r"\n{2,}"):
    """ Returns a list of sentences. Each sentence is a space-separated string of tokens (words).
        Handles common cases of abbreviations (e.g., etc., ...).
        Punctuation marks are split from other words. Periods (or ?!) mark the end of a sentence.
        Headings without an ending period are inferred by line breaks.
    """
    # Handle periods separately.
    punctuation = tuple(punctuation.replace(".", ""))
    # Handle replacements (contractions).
    for a, b in replace.items():
        string = re.sub(a, b, string)
    # Handle Unicode quotes.
    if isinstance(string, unicode):
        string = string.replace(u"“", u" “ ")
        string = string.replace(u"”", u" ” ")
        string = string.replace(u"‘", u" ‘ ")
        string = string.replace(u"’", u" ’ ")
    # Collapse whitespace.
    string = re.sub("\r\n", "\n", string)
    string = re.sub(linebreak, " %s " % EOS, string)
    string = re.sub(r"\s+", " ", string)
    tokens = []
    # Handle punctuation marks.
    for t in TOKEN.findall(string+" "):
        if len(t) > 0:
            tail = []
            while t.startswith(punctuation) and \
              not t in replace:
                # Split leading punctuation.
                if t.startswith(punctuation):
                    tokens.append(t[0]); t=t[1:]
            while t.endswith(punctuation+(".",)) and \
              not t in replace:
                # Split trailing punctuation.
                if t.endswith(punctuation):
                    tail.append(t[-1]); t=t[:-1]
                # Split ellipsis (...) before splitting period.
                if t.endswith("..."):
                    tail.append("..."); t=t[:-3].rstrip(".")
                # Split period (if not an abbreviation).
                if t.endswith("."):
                    if t in abbreviations or \
                      RE_ABBR1.match(t) is not None or \
                      RE_ABBR2.match(t) is not None or \
                      RE_ABBR3.match(t) is not None:
                        break
                    else:
                        tail.append(t[-1]); t=t[:-1]
            if t != "":
                tokens.append(t)
            tokens.extend(reversed(tail))
    # Handle sentence breaks (periods, quotes, parenthesis).
    sentences, i, j = [[]], 0, 0
    while j < len(tokens):
        if tokens[j] in ("...", ".", "!", "?", EOS):
            while j < len(tokens) \
              and tokens[j] in ("'", "\"", u"”", u"’", "...", ".", "!", "?", ")", EOS):
                if tokens[j] in ("'", "\"") and sentences[-1].count(tokens[j]) % 2 == 0:
                    break # Balanced quotes.
                j += 1
            sentences[-1].extend(t for t in tokens[i:j] if t != EOS)
            sentences.append([])
            i = j
        j += 1
    # Handle emoticons.
    sentences[-1].extend(tokens[i:j])
    sentences = (" ".join(s) for s in sentences if len(s) > 0)
    sentences = (RE_SARCASM.sub("(!)", s) for s in sentences)
    sentences = [RE_EMOTICONS.sub(
        lambda m: m.group(1).replace(" ", "") + m.group(2), s) for s in sentences]
    return sentences