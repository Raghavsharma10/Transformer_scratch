def find(path,
         level=None,
         message=None,
         time_lower=None, time_upper=None,
         case_sensitive=False):  # pragma: no cover
    """
    Filter log message.

    **中文文档**

    根据level名称, message中的关键字, 和log的时间的区间, 筛选出相关的日志
    """
    if level:
        level = level.upper()  # level name has to be capitalized.

    if not case_sensitive:
        message = message.lower()

    with open(path, "r") as f:
        result = Result(path=path,
                        level=level, message=message,
                        time_lower=time_lower, time_upper=time_upper,
                        case_sensitive=case_sensitive,
                        )

        for line in f:
            try:
                _time, _level, _message = [i.strip() for i in line.split(";")]

                if level:
                    if _level != level:
                        continue

                if time_lower:
                    if _time < time_lower:
                        continue

                if time_upper:
                    if _time > time_upper:
                        continue

                if message:
                    if not case_sensitive:
                        _message = _message.lower()

                    if message not in _message:
                        continue

                result.lines.append(line)
            except Exception as e:
                print(e)

    return result