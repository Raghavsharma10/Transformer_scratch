def register(self, *args, **kwargs):
        '''
        调用方可根据主键字段进行爬虫的创建或更新操作

        :return: 返回符合接口定义的字典数据
        :rtype: dict
        '''
        return {
            'name': zhihu.name,
            'display_name': zhihu.display_name,
            'author': zhihu.author,
            'email': zhihu.email,
            'description': zhihu.description,
            'meta': {
                # 爬取计划，参考 crontab 配置方法
                'crawl_schedule': '0 23 * * *',

                # 执行爬取的随机延时，单位秒，用于避免被 Ban
                'crawl_random_delay': str(60 * 60),

                'package_module': 'mobi',
                'language': 'zh-CN',
                'book_mode': 'periodical',  # 'periodical' | 'book'
                'img_cover': os.path.join(
                    _images_path, 'cv_zhihudaily.jpg'),
                'img_masthead': os.path.join(
                    _images_path, 'mh_zhihudaily.gif'),
                'image_filter': json.dumps(['zhihu.com/equation']),
                'css_package': os.path.join(
                    _css_path, 'package.css')
            }
        }