def build_fasttext_wiki_embedding_obj(embedding_type):
    """FastText pre-trained word vectors for 294 languages, with 300 dimensions, trained on Wikipedia. It's recommended to use the same tokenizer for your data that was used to construct the embeddings. It's implemented as 'FasttextWikiTokenizer'. More information: https://fasttext.cc/docs/en/pretrained-vectors.html.

    Args:
        embedding_type: A string in the format `fastext.wiki.$LANG_CODE`. e.g. `fasttext.wiki.de` or `fasttext.wiki.es`
    Returns:
        Object with the URL and filename used later on for downloading the file.
    """
    lang = embedding_type.split('.')[2]
    return {
        'file': 'wiki.{}.vec'.format(lang),
        'url': 'https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.{}.vec'.format(lang),
        'extract': False,
    }