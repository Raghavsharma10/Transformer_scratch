def minimum_entropy_match_sequence(password, matches):
    """
    Returns minimum entropy

    Takes a list of overlapping matches, returns the non-overlapping sublist with
    minimum entropy. O(nm) dp alg for length-n password with m candidate matches.
    """
    bruteforce_cardinality = calc_bruteforce_cardinality(password) # e.g. 26 for lowercase
    up_to_k = [0] * len(password) # minimum entropy up to k.
    # for the optimal sequence of matches up to k, holds the final match (match['j'] == k). null means the sequence ends
    # without a brute-force character.
    backpointers = []
    for k in range(0, len(password)):
        # starting scenario to try and beat: adding a brute-force character to the minimum entropy sequence at k-1.
        up_to_k[k] = get(up_to_k, k-1) + lg(bruteforce_cardinality)
        backpointers.append(None)
        for match in matches:
            if match['j'] != k:
                continue
            i, j = match['i'], match['j']
            # see if best entropy up to i-1 + entropy of this match is less than the current minimum at j.
            up_to = get(up_to_k, i-1)
            candidate_entropy = up_to + calc_entropy(match)
            if candidate_entropy < up_to_k[j]:
                #print "New minimum: using " + str(match)
                #print "Entropy: " + str(candidate_entropy)
                up_to_k[j] = candidate_entropy
                backpointers[j] = match

    # walk backwards and decode the best sequence
    match_sequence = []
    k = len(password) - 1
    while k >= 0:
        match = backpointers[k]
        if match:
            match_sequence.append(match)
            k = match['i'] - 1
        else:
            k -= 1
    match_sequence.reverse()

    # fill in the blanks between pattern matches with bruteforce "matches"
    # that way the match sequence fully covers the password: match1.j == match2.i - 1 for every adjacent match1, match2.
    def make_bruteforce_match(i, j):
        return {
            'pattern': 'bruteforce',
            'i': i,
            'j': j,
            'token': password[i:j+1],
            'entropy': lg(math.pow(bruteforce_cardinality, j - i + 1)),
            'cardinality': bruteforce_cardinality,
        }
    k = 0
    match_sequence_copy = []
    for match in match_sequence:
        i, j = match['i'], match['j']
        if i - k > 0:
            match_sequence_copy.append(make_bruteforce_match(k, i - 1))
        k = j + 1
        match_sequence_copy.append(match)

    if k < len(password):
        match_sequence_copy.append(make_bruteforce_match(k, len(password) - 1))
    match_sequence = match_sequence_copy

    min_entropy = 0 if len(password) == 0 else up_to_k[len(password) - 1] # corner case is for an empty password ''
    crack_time = entropy_to_crack_time(min_entropy)

    # final result object
    return {
        'password': password,
        'entropy': round_to_x_digits(min_entropy, 3),
        'match_sequence': match_sequence,
        'crack_time': round_to_x_digits(crack_time, 3),
        'crack_time_display': display_time(crack_time),
        'score': crack_time_to_score(crack_time),
    }