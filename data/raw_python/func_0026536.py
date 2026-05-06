def print_messages(domain, msg):
    """Debugging function to print all message language variants"""

    domain = Domain(domain)
    for lang in all_languages():
        print(lang, ':', domain.get(lang, msg))