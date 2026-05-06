def _tag_most_likely(examples):
    """
    Return a list of date elements by choosing the most likely element for a token within examples (context-free).
    """
    tokenized_examples = [_tokenize_by_character_class(example) for example in examples]

    # We currently need the tokenized_examples to all have the same length, so drop instances that have a length
    # that does not equal the mode of lengths within tokenized_examples
    token_lengths = [len(e) for e in tokenized_examples]
    token_lengths_mode = _mode(token_lengths)
    tokenized_examples = [example for example in tokenized_examples if len(example) == token_lengths_mode]

    # Now, we iterate through the tokens, assigning date elements based on their likelihood. In cases where
    # the assignments are unlikely for all date elements, assign filler.
    most_likely = []
    for token_index in range(0, token_lengths_mode):
        tokens = [token[token_index] for token in tokenized_examples]
        probabilities = _percent_match(DATE_ELEMENTS, tokens)
        max_prob = max(probabilities)
        if max_prob < 0.5:
            most_likely.append(Filler(_mode(tokens)))
        else:
            if probabilities.count(max_prob) == 1:
                most_likely.append(DATE_ELEMENTS[probabilities.index(max_prob)])
            else:
                choices = []
                for index, prob in enumerate(probabilities):
                    if prob == max_prob:
                        choices.append(DATE_ELEMENTS[index])
                most_likely.append(_most_restrictive(choices))

    return most_likely