def main():
    """
    main
    """
    try:
        sr71 = BlackBird()
        sr71.start()
    except BlackbirdError as error:
        sys.stderr.write(error.__str__() + '\n')
        return(1)