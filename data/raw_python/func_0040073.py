def smart_insert(df, table, engine, minimal_size=5):
    """An optimized Insert strategy.

    **中文文档**

    一种优化的将大型DataFrame中的数据, 在有IntegrityError的情况下将所有
    好数据存入数据库的方法。
    """
    from sqlalchemy.exc import IntegrityError

    try:
        table_name = table.name
    except:
        table_name = table

    # 首先进行尝试bulk insert
    try:
        df.to_sql(table_name, engine, index=False, if_exists="append")
    # 失败了
    except IntegrityError:
        # 分析数据量
        n = df.shape[0]
        # 如果数据条数多于一定数量
        if n >= minimal_size ** 2:
            # 则进行分包
            n_chunk = math.floor(math.sqrt(n))
            for sub_df in grouper_df(df, n_chunk):
                smart_insert(
                    sub_df, table_name, engine, minimal_size)
        # 否则则一条条地逐条插入
        else:
            for sub_df in grouper_df(df, 1):
                try:
                    sub_df.to_sql(
                        table_name, engine, index=False, if_exists="append")
                except IntegrityError:
                    pass