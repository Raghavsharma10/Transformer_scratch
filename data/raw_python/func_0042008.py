def closed(self, reason):
        '''
        异步爬取本地化处理完成后，使用结果数据，进行输出文件的渲染，渲染完毕，
        调用 :meth:`.MobiSpider.generate_mobi_file` 方法，生成目标 ``mobi`` 文件
        '''
        # 拷贝封面&报头图片文件
        utils.mkdirp(os.path.join(self.build_source_dir, 'images'))
        self._logger.info(self.options)
        shutil.copy(
            self.options.get('img_cover'),
            os.path.join(self.build_source_dir, 'images', 'cover.jpg'))
        shutil.copy(
            self.options.get('img_masthead'),
            os.path.join(self.build_source_dir, 'images', 'masthead.gif'))

        # 拷贝css文件
        css_base_path = self.options.get('css_base')
        css_package_path = self.options.get('css_package')
        css_extra = self.options.get('extra_css', '')
        css_output_dir = os.path.join(self.build_source_dir, 'css')
        utils.mkdirp(css_output_dir)
        if css_base_path:
            shutil.copy(
                css_base_path,
                os.path.join(css_output_dir, 'base.css'))
        if css_package_path:
            shutil.copy(
                css_package_path,
                os.path.join(css_output_dir, 'package.css'))
        if css_extra:
            with codecs.open(
                    os.path.join(css_output_dir, 'custom.css'),
                    'wb', 'utf-8') as fh:
                fh.write(css_extra)

        # 拷贝icons路径文件
        icons_path = self.options.get('icons_path')
        icons_output_dir = os.path.join(self.build_source_dir, 'icons')
        shutil.rmtree(icons_output_dir, ignore_errors=True)
        if icons_path:
            shutil.copytree(icons_path, icons_output_dir)

        # 获取content模板对象
        template_content_path = os.path.join(
            self.template_dir, 'OEBPS', 'content.opf')
        with open(template_content_path, 'r') as fh:
            template_content = Template(fh.read())

        # 渲染content目标文件
        content_path = os.path.join(self.build_source_dir, 'moear.opf')
        with codecs.open(content_path, 'wb', 'utf-8') as fh:
            fh.write(template_content.render(
                data=self.data,
                spider=self.spider,
                options=self.options))

        # 获取toc.ncx模板对象
        template_toc_path = os.path.join(
            self.template_dir, 'OEBPS', 'toc.ncx')
        with open(template_toc_path, 'r') as fh:
            template_toc = Template(fh.read())

        # 渲染toc.ncx目标文件
        toc_path = os.path.join(self.build_source_dir, 'misc', 'toc.ncx')
        utils.mkdirp(os.path.dirname(toc_path))
        with codecs.open(toc_path, 'wb', 'utf-8') as fh:
            fh.write(template_toc.render(
                data=self.data,
                spider=self.spider,
                options=self.options))

        # 获取toc.html模板对象
        template_toc_path = os.path.join(
            self.template_dir, 'OEBPS', 'toc.html')
        with open(template_toc_path, 'r') as fh:
            template_toc = Template(fh.read())

        # 渲染toc.html目标文件
        toc_path = os.path.join(self.build_source_dir, 'html', 'toc.html')
        utils.mkdirp(os.path.dirname(toc_path))
        with codecs.open(toc_path, 'wb', 'utf-8') as fh:
            fh.write(template_toc.render(
                data=self.data,
                options=self.options))

        # 生成mobi文件到mobi_dir
        self.generate_mobi_file()