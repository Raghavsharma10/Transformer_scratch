def format(self, data, *args, **kwargs):
        '''
        将传入的Post列表数据进行格式化处理。此处传入的 ``data`` 格式即为
        :meth:`.ZhihuDaily.crawl` 返回的格式，但具体内容可以不同，即此处保留了灵活度，
        可以对非当日文章对象进行格式化，制作相关主题的合集书籍

        :param data: 待处理的文章列表
        :type data: list

        :return: 返回符合mobi打包需求的定制化数据结构
        :rtype: dict
        '''
        sections = OrderedDict()
        hot_list = []
        normal_list = []
        for item in data:
            meta = item.get('meta', [])

            # 如果标题为空，则迭代下一条目
            if not item.get('title'):
                continue

            soup = BeautifulSoup(item.get('content'), "lxml")

            # 清洗文章内容，去除无用内容
            for view_more in soup.select('.view-more'):
                view_more.extract()
            item['content'] = str(soup.div)

            # 处理文章摘要，若为空则根据正文自动生成并填充
            if not item.get('excerpt') and item.get('content'):
                word_limit = self.options.get(
                    'toc_desc_word_limit', 500)
                content_list = soup.select('div.content')
                content_list = [content.get_text() for content in content_list]
                excerpt = ' '.join(content_list)[:word_limit]
                # 此处摘要信息需进行HTML转义，否则会造成toc.ncx中tag处理错误
                item['excerpt'] = html.escape(excerpt)

            # 从item中提取出section分组
            top = meta.pop('spider.zhihu_daily.top', '0')
            item['meta'] = meta
            if str(top) == '1':
                hot_list.append(item)
            else:
                normal_list.append(item)

        if hot_list:
            sections.setdefault('热闻', hot_list)
        if normal_list:
            sections.setdefault('日报', normal_list)
        return sections