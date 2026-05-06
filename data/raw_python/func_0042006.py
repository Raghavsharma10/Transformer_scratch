def filter_images_urls(image_urls, image_filter, common_image_filter=None):
        '''
        图片链接过滤器，根据传入的过滤器规则，对图片链接列表进行过滤并返回结果列表

        :param list(str) image_urls: 图片链接字串列表
        :param list(str) image_filter: 过滤器字串列表
        :param list(str) common_image_filter: 可选，通用的基础过滤器，
            会在定制过滤器前对传入图片应用
        :return: 过滤后的结果链接列表，以及被过滤掉的链接列表
        :rtype: list(str), list(str)
        :raises TypeError: image_filter 不为字串或列表
        :raises ValueError: image_filter 中存在空值
        '''
        common_image_filter = common_image_filter or []

        # 对图片过滤器进行完整性验证
        image_filter = json.loads(image_filter, encoding='utf-8')
        if not isinstance(image_filter, (str, list)):
            raise TypeError('image_filter not str or list')
        if isinstance(image_filter, str):
            image_filter = [image_filter]
        if not all(image_filter):
            raise ValueError('image_filter 中存在空值：{}'.format(image_filter))

        rc = copy.deepcopy(image_urls)
        rc_removed = []
        for i in image_urls:
            # 执行内置过滤器
            for f in common_image_filter:
                if re.search(f, i):
                    rc.remove(i)
                    rc_removed.append(i)

            # 执行具体文章源的定制过滤器
            for f in image_filter:
                if re.search(f, i):
                    rc.remove(i)
                    rc_removed.append(i)
        return rc, rc_removed