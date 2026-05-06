def is_rhyme(d, w1, w2):
    """check if words rhyme"""
    for p1 in d[w1]:
        # extract only "rhyming portion"
        p1 = p1.split("'")[-1]
        m = VOWELS_RE.search(p1)
        if not m:
            print(p1)
        p1 = p1[m.start():]
        for p2 in d[w2]:
            p2 = p2.split("'")[-1]
            m = VOWELS_RE.search(p2)
            if not m:
                print(w2, p2)
            p2 = p2[m.start():]
            if p1 == p2:
                return True
    return False