def item_completed(self, results, item, info):
        '''
        在正常图片本地化处理管道业务执行完毕后，使用缩略图路径替换原 ``result[path]`` 路径，
        从而使最终打包时使用缩略图，并根据配置，对缩略图进行灰度处理

        :param item: 爬取到的数据模型
        :type item: :class:`.MoearPackageMobiItem` or dict
        '''
        # 处理 results 中的 path 使用缩略图路径替代
        for ok, result in results:
            if not ok:
                continue
            path = result['path']
            path = re.sub(r'full', os.path.join('thumbs', 'kindle'), path)
            result['path'] = path

        # 处理缩略图为灰度图，为便于在电纸书上节省空间
        if info.spider.options.get('img_convert_to_gray'):
            images_store = info.spider.settings.get('IMAGES_STORE')
            for ok, result in results:
                if not ok:
                    continue

                img_path = os.path.join(images_store, result['path'])
                with open(img_path, 'rb+') as fh:
                    output = img.gray_image(fh.read())
                    fh.seek(0)
                    fh.truncate()
                    fh.write(output)

        info.spider._logger.debug(results)
        item = super(MoEarImagesPipeline, self).item_completed(
            results, item, info)
        return item