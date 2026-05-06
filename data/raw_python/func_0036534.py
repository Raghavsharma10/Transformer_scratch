def _default_tokenizer_func(content, keyword_weight_pair):
    """
    Default tokenizer function that uses jieba tokenizer.

    :param keyword_weight_pair: maximum pair number of the keyword-weight list.
    :return: return keyword-weight list. Example: [('Example',0.4511233019962264),('Hello',0.25548051420382073),...].
    """
    seg_list = jieba.lcut_for_search(content)
    # Extract keyword-weight list by TF-IDF algorithms and by sorted maximum weight
    return jieba.analyse.extract_tags("".join(seg_list), topK=keyword_weight_pair, withWeight=True)