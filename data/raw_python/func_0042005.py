def parse(self, response):
        """
        从 self.data 中将文章信息格式化为 :class:`.MoearPackageMobiItem`
        """
        # 工作&输出路径
        self.template_dir = self.settings.get('TEMPLATE_DIR')
        shutil.rmtree(
            self.settings.get('BUILD_SOURCE_DIR'), ignore_errors=True)
        self.build_source_dir = utils.mkdirp(
            self.settings.get('BUILD_SOURCE_DIR'))

        # 获取Post模板对象
        template_post_path = os.path.join(self.template_dir, 'post.html')
        with open(template_post_path, 'r') as f:
            self.template_post = Template(f.read())

        self._logger.info('构建处理路径 => {0}'.format(self.build_source_dir))

        image_filter = self.options.get('image_filter', '')
        common_image_filter = self.options.get('common_image_filter', [])
        for sections in self.data.values():
            for p in sections:
                item = MoearPackageMobiItem()
                pmeta = p.get('meta', {})
                item['url'] = p.get('origin_url', '')
                item['title'] = p.get('title', '')
                item['cover_image'] = pmeta.get('moear.cover_image_slug')
                item['content'] = p.get('content', '')

                # 为图片持久化pipeline执行做数据准备
                item['image_urls'] = [item['cover_image']] \
                    if item['cover_image'] is not None else []
                item['image_urls'] += \
                    self._populated_image_urls_with_content(item['content'])
                self._logger.debug(
                    '待处理的图片url(过滤前): {}'.format(item['image_urls']))
                item['image_urls'], item['image_urls_removed'] = \
                    self.filter_images_urls(
                        item['image_urls'], image_filter, common_image_filter)
                self._logger.debug(
                    '待处理的图片url: {}'.format(item['image_urls']))

                yield item