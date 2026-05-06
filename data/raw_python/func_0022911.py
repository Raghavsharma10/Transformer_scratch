def validate_input_format(utterance, intent):
    """ TODO add handling for bad input"""
    slots = {slot["name"] for slot in intent["slots"]}
    split_utt = re.split("{(.*)}", utterance)
    banned = set("-/\\()^%$#@~`-_=+><;:") # Banned characters

    for token in split_utt:
        if (banned & set(token)):
            print (" - Banned character found in substring", token)
            print (" - Banned character list", banned)
            return False

        if "|" in token:
            split_token = token.split("|")
            if len(split_token)!=2:
                print (" - Error, token is incorrect in", token, split_token)
                return False

            word, slot = split_token
            if slot.strip() not in slots:
                print (" -", slot, "is not a valid slot for this Intent, valid slots are", slots)
                return False
    return True