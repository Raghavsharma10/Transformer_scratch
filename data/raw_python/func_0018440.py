def latex_to_img(tex):
        """Return a pygame image from a latex template."""
        with tempfile.TemporaryDirectory() as tmpdirname:
            with open(tmpdirname + r'\tex.tex', 'w') as f:
                f.write(tex)

            os.system(r"latex {0}\tex.tex -halt-on-error -interaction=batchmode -disable-installer -aux-directory={0} "
                      r"-output-directory={0}".format(tmpdirname))
            os.system(r"dvipng -T tight -z 9 --truecolor -o {0}\tex.png {0}\tex.dvi".format(tmpdirname))
            # os.system(r'latex2png ' + tmpdirname)

            image = pygame.image.load(tmpdirname + r'\tex.png')

        return image