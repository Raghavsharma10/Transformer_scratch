def crawl(self, *args, **kwargs):
        '''
        执行爬取操作，并阻塞直到爬取完成，返回结果数据。
        此处考虑到 Scrapy 本身的并发特性，故通过临时文件方式做数据传递，
        将临时路径传递到爬虫业务中，并在爬取结束后对文件进行读取、 JSON 反序列化，返回

        :return: 返回符合接口定义的字典对象
        :rtype: dict
        '''
        temp = tempfile.NamedTemporaryFile(mode='w+t')

        try:
            crawler = CrawlerScript()
            # 调试时可指定明确日期参数，如：date='20180423'
            crawler.crawl(output_file=temp.name, *args, **kwargs)

            temp.seek(0)
            content = json.loads(temp.read(), encoding='UTF-8')
        finally:
            temp.close()

        print('抓取完毕！')
        return content