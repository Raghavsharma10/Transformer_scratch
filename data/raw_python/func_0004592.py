def build_language(source_code: str, lang: Language, filename: str):
    """
    lang:      language object represents your language.
    """
    state = MetaState(rbnf.implementation, requires=_Wild(), filename=filename)
    state.data = lang
    _build_language(source_code, state)
    lang.build()