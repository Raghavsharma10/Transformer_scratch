def closed(self, reason):
        '''
        异步爬取全部结束后，执行此关闭方法，对 ``item_list`` 中的数据进行 **JSON**
        序列化，并输出到指定文件中，传递给 :meth:`.ZhihuDaily.crawl`

        :param obj reason: 爬虫关闭原因
        '''
        self.logger.debug('结果列表: {}'.format(self.item_list))

        output_strings = json.dumps(self.item_list, ensure_ascii=False)
        with open(self.output_file, 'w') as fh:
            fh.write(output_strings)