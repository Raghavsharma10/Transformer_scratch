def build_list_result(results, xml):
    """
    构建带翻页的列表
    
    :param results: 已获取的数据列表
    :param xml: 原始页面xml
    :return: {'results': list, 'count': int, 'next_start': int|None}
            如果count与results长度不同，则有更多
            如果next_start不为None，则可以到下一页
    """
    xml_count = xml.xpath('//div[@class="paginator"]/span[@class="count"]/text()')
    xml_next = xml.xpath('//div[@class="paginator"]/span[@class="next"]/a/@href')
    count = int(re.search(r'\d+', xml_count[0]).group()) if xml_count else len(results)
    next_start = int(re.search(r'start=(\d+)', xml_next[0]).groups()[0]) if xml_next else None
    return {'results': results, 'count': count, 'next_start': next_start}