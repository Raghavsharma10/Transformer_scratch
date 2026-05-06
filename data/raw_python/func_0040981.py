def generate(self, data, *args, **kwargs):
        """
        根据传入的数据结构生成最终用于推送的文件字节字符串( :func:`bytes` )，
        MoEar会将其持久化并用于之后的推送任务

        :param dict data: 待打包的数据结构
        :return: 返回生成的书籍打包输出字节
        :rtype: bytes
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.options.setdefault('package_build_dir', tmpdirname)
            crawler = CrawlerScript(self.options)
            crawler.crawl(data, self.spider, *args, **kwargs)

            output_file = os.path.join(
                self.options['package_build_dir'], 'source', 'moear.mobi')
            with open(output_file, 'rb') as fh:
                content = fh.read()

        return content