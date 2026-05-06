def print_tree(sent, token_attr):
    """Prints sentences tree as string using token_attr from token(like pos_, tag_ etc.)

        :param sent: sentence to print
        :param token_attr: choosen attr to present for tokens(e.g. dep_, pos_, tag_, ...)

    """
    def __print_sent__(token, attr):
        print("{", end=" ")
        [__print_sent__(t, attr) for t in token.lefts]
        print(u"%s->%s(%s)" % (token,token.dep_,token.tag_ if not attr else getattr(token, attr)), end="")
        [__print_sent__(t, attr) for t in token.rights]
        print("}", end=" ")
    return __print_sent__(sent.root, token_attr)