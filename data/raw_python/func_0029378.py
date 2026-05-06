def ask_captcha(length=4):
    """Prompts the user for a random string."""
    captcha = "".join(random.choice(string.ascii_lowercase) for _ in range(length))
    ask_str('Enter the following letters, "%s"' % (captcha), vld=[captcha, captcha.upper()], blk=False)