def parse_tr_strs(trs):
    """
    将没有值但有必须要的单元格的值设置为 None
    将 <tr> 标签数组内的单元格文字解析出来并返回一个二维列表

    :param trs: <tr> 标签或标签数组, 为 :class:`bs4.element.Tag` 对象
    :return: 二维列表
    """
    tr_strs = []
    for tr in trs:
        strs = []
        for td in tr.find_all('td'):
            text = td.get_text(strip=True)
            strs.append(text or None)
        tr_strs.append(strs)
    logger.debug('从行中解析出以下数据\n%s', pformat(tr_strs))
    return tr_strs