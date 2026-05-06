def get_token_stream(source: str) -> CommonTokenStream:
    """ Get the antlr token stream.
    """
    lexer = LuaLexer(InputStream(source))
    stream = CommonTokenStream(lexer)
    return stream