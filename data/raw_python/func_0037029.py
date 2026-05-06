def check_updates():
    """Check and display upgraded packages
    """
    count, packages = fetch()
    message = "No news is good news !"
    if count > 0:
        message = ("{0} software updates are available\n".format(count))
    return [message, count, packages]