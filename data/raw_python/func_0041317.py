def random_sample(col, n=5, filters=None):
    """Randomly select n document from query result set. If no query specified,
    then from entire collection.

    **中文文档**

    从collection中随机选择 ``n`` 个样本。
    """
    pipeline = list()
    if filters is not None:
        pipeline.append({"$match": filters})
    pipeline.append({"$sample": {"size": n}})
    return list(col.aggregate(pipeline))