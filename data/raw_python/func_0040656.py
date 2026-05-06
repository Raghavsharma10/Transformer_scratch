def parse_post(self, response):
        '''
        根据 :meth:`.ZhihuDailySpider.parse` 中生成的具体文章地址，获取到文章内容，
        并对其进行格式化处理，结果填充到对象属性 ``item_list`` 中

        :param Response response: 由 ``Scrapy`` 调用并传入的请求响应对象
        '''
        content = json.loads(response.body.decode(), encoding='UTF-8')
        post = response.meta['post']

        post['origin_url'] = content.get('share_url', '')
        if not all([post['origin_url']]):
            raise ValueError('原文地址为空')

        post['title'] = html.escape(content.get('title', ''))
        if not all([post['title']]):
            raise ValueError('文章标题为空 - {}'.format(post.get('origin_url')))

        # 单独处理type字段为1的情况，即该文章为站外转发文章
        if content.get('type') == 1:
            self.logger.warn('遇到站外文章，单独处理 - {}'.format(post['title']))
            return post

        soup = BeautifulSoup(content.get('body', ''), 'lxml')
        author_obj = soup.select('span.author')
        self.logger.debug(author_obj)
        if author_obj:
            author_list = []
            for author in author_obj:
                author_list.append(
                    author.string.rstrip('，, ').replace('，', ','))
            author_list = list(set(author_list))
            post['author'] = html.escape('，'.join(author_list))
        post['content'] = str(soup.div)

        # 继续填充post数据
        image_back = content.get('images', [None])[0]
        if image_back:
            post['meta']['moear.cover_image_slug'] = \
                content.get('image', image_back)
        self.logger.debug(post)