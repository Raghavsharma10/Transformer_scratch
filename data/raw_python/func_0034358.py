def create_lines(self, draw, n_line, width, height):
        '''绘制干扰线'''
        line_num = randint(n_line[0], n_line[1])  # 干扰线条数
        for i in range(line_num):
            # 起始点
            begin = (randint(0, width), randint(0, height))
            # 结束点
            end = (randint(0, width), randint(0, height))
            draw.line([begin, end], fill=(0, 0, 0))