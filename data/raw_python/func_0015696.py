def lookup_name_fast(self, name):
        """Might return a struct"""

        if name in self.__names:
            return self.__names[name]

        count = self.__get_count_cached()
        lo = 0
        hi = count
        while lo < hi:
            mid = (lo + hi) // 2
            if self.__get_name_cached(mid) < name:
                lo = mid + 1
            else:
                hi = mid

        if lo != count and self.__get_name_cached(lo) == name:
            return self.__get_info_cached(lo)