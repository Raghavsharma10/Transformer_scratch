def compile_regex(self, fmt, query):
        """Turn glob (graphite) queries into compiled regex
        * becomes .*
        . becomes \.
        fmt argument is so that caller can control anchoring (must contain exactly 1 {0} !"""
        return re.compile(fmt.format(
            query.pattern.replace('.', '\.').replace('*', '[^\.]*').replace(
                '{', '(').replace(',', '|').replace('}', ')')
        ))