def thumbnail(self, remote_path, height, width, quality=100, **kwargs):
        """获取指定图片文件的缩略图.

        :param remote_path: 源图片的路径，路径必须以 /apps/ 开头。

                            .. warning::
                                * 路径长度限制为1000；
                                * 径中不能包含以下字符：``\\\\ ? | " > < : *``；
                                * 文件名或路径名开头结尾不能是 ``.``
                                  或空白字符，空白字符包括：
                                  ``\\r, \\n, \\t, 空格, \\0, \\x0B`` 。
        :param height: 指定缩略图的高度，取值范围为(0,1600]。
        :type height: int
        :param width: 指定缩略图的宽度，取值范围为(0,1600]。
        :type width: int
        :param quality: 缩略图的质量，默认为100，取值范围(0,100]。
        :type quality: int
        :return: Response 对象

        .. warning::
           有以下限制条件：

           * 原图大小(0, 10M]；
           * 原图类型: jpg、jpeg、bmp、gif、png；
           * 目标图类型:和原图的类型有关；例如：原图是gif图片，
             则缩略后也为gif图片。
        """

        params = {
            'path': remote_path,
            'height': height,
            'width': width,
            'quality': quality,
        }
        return self._request('thumbnail', 'generate', extra_params=params,
                             **kwargs)