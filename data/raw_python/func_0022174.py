def _create_image(self, matrix, width, height, pad):
        """
        Generates a PNG byte list
        """

        image = Image.new("RGB", (width + (pad * 2),
                                  height + (pad * 2)), self.bg_colour)
        image_draw = ImageDraw.Draw(image)

        # Calculate the block width and height.
        block_width = float(width) / self.cols
        block_height = float(height) / self.rows

        # Loop through blocks in matrix, draw rectangles.
        for row, cols in enumerate(matrix):
            for col, cell in enumerate(cols):
                if cell:
                    image_draw.rectangle((
                        pad + col * block_width,  # x1
                        pad + row * block_height,  # y1
                        pad + (col + 1) * block_width - 1,  # x2
                        pad + (row + 1) * block_height - 1  # y2
                    ), fill=self.fg_colour)

        stream = BytesIO()
        image.save(stream, format="png", optimize=True)
        # return the image byte data
        return stream.getvalue()