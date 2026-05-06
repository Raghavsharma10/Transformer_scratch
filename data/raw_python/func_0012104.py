def make_basename(self, fn=None, ext=None):
        """make a filesystem-compliant basename for this file"""
        fb, oldext = os.path.splitext(os.path.basename(fn or self.fn))
        ext = ext or oldext.lower()
        fb = String(fb).hyphenify(ascii=True)
        return ''.join([fb, ext])