def build_fasttext_cc_embedding_obj(embedding_type):
    """FastText pre-trained word vectors for 157 languages, with 300 dimensions, trained on Common Crawl and Wikipedia. Released in 2018, it succeesed the 2017 FastText Wikipedia embeddings. It's recommended to use the same tokenizer for your data that was used to construct the embeddings. This information and more can be find on their Website: https://fasttext.cc/docs/en/crawl-vectors.html.

    Args:
        embedding_type: A string in the format `fastext.cc.$LANG_CODE`. e.g. `fasttext.cc.de` or `fasttext.cc.es`
    Returns:
        Object with the URL and filename used later on for downloading the file.
    """
    lang = embedding_type.split('.')[2]
    return {
        'file': 'cc.{}.300.vec.gz'.format(lang),
        'url': 'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.{}.300.vec.gz'.format(lang),
        'extract': False
    }