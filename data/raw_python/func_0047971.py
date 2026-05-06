def convert_nonascii(self, lst):
        """Convert the strange outputs from git commands"""
        for item in lst:
            if item.startswith('"') and item.endswith('"'):
                item = item[1:-1]
                yield item.encode('utf-8').decode('unicode-escape')
            else:
                yield item.encode('utf-8').decode('unicode-escape')