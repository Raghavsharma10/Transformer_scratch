def to_meshcode(lat, lon, level):
    """緯度経度から指定次の地域メッシュコードを算出する。

    Args:
        lat: 世界測地系の緯度(度単位)
        lon: 世界測地系の経度(度単位)
        level: 地域メッシュコードの次数
                1次(80km四方):1
                40倍(40km四方):40000
                20倍(20km四方):20000
                16倍(16km四方):16000
                2次(10km四方):2
                8倍(8km四方):8000
                5倍(5km四方):5000
                4倍(4km四方):4000
                2.5倍(2.5km四方):2500
                2倍(2km四方):2000
                3次(1km四方):3
                4次(500m四方):4
                5次(250m四方):5
                6次(125m四方):6
    Return:
        指定次の地域メッシュコード

    """

    if not 0 <= lat < 66.66:
        raise ValueError('the latitude is out of bound.')

    if not 100 <= lon < 180:
        raise ValueError('the longitude is out of bound.')

    # reminder of latitude and longitude by its unit in degree of mesh level.
    rem_lat_lv0 = lambda lat: lat
    rem_lon_lv0 = lambda lon: lon % 100
    rem_lat_lv1 = lambda lat: rem_lat_lv0(lat) % _unit_lat_lv1()
    rem_lon_lv1 = lambda lon: rem_lon_lv0(lon) % _unit_lon_lv1()
    rem_lat_40000 = lambda lat: rem_lat_lv1(lat) % _unit_lat_40000()
    rem_lon_40000 = lambda lon: rem_lon_lv1(lon) % _unit_lon_40000()
    rem_lat_20000 = lambda lat: rem_lat_40000(lat) % _unit_lat_20000()
    rem_lon_20000 = lambda lon: rem_lon_40000(lon) % _unit_lon_20000()
    rem_lat_16000 = lambda lat: rem_lat_lv1(lat) % _unit_lat_16000()
    rem_lon_16000 = lambda lon: rem_lon_lv1(lon) % _unit_lon_16000()
    rem_lat_lv2 = lambda lat: rem_lat_lv1(lat) % _unit_lat_lv2()
    rem_lon_lv2 = lambda lon: rem_lon_lv1(lon) % _unit_lon_lv2()
    rem_lat_8000 = lambda lat: rem_lat_lv1(lat) % _unit_lat_8000()
    rem_lon_8000 = lambda lon: rem_lon_lv1(lon) % _unit_lon_8000()
    rem_lat_5000 = lambda lat: rem_lat_lv2(lat) % _unit_lat_5000()
    rem_lon_5000 = lambda lon: rem_lon_lv2(lon) % _unit_lon_5000()
    rem_lat_4000 = lambda lat: rem_lat_8000(lat) % _unit_lat_4000()
    rem_lon_4000 = lambda lon: rem_lon_8000(lon) % _unit_lon_4000()
    rem_lat_2500 = lambda lat: rem_lat_5000(lat) % _unit_lat_2500()
    rem_lon_2500 = lambda lon: rem_lon_5000(lon) % _unit_lon_2500()
    rem_lat_2000 = lambda lat: rem_lat_lv2(lat) % _unit_lat_2000()
    rem_lon_2000 = lambda lon: rem_lon_lv2(lon) % _unit_lon_2000()
    rem_lat_lv3 = lambda lat: rem_lat_lv2(lat) % _unit_lat_lv3()
    rem_lon_lv3 = lambda lon: rem_lon_lv2(lon) % _unit_lon_lv3()
    rem_lat_lv4 = lambda lat: rem_lat_lv3(lat) % _unit_lat_lv4()
    rem_lon_lv4 = lambda lon: rem_lon_lv3(lon) % _unit_lon_lv4()
    rem_lat_lv5 = lambda lat: rem_lat_lv4(lat) % _unit_lat_lv5()
    rem_lon_lv5 = lambda lon: rem_lon_lv4(lon) % _unit_lon_lv5()
    rem_lat_lv6 = lambda lat: rem_lat_lv5(lat) % _unit_lat_lv6()
    rem_lon_lv6 = lambda lon: rem_lon_lv5(lon) % _unit_lon_lv6()

    def meshcode_lv1(lat, lon):
        ab = int(rem_lat_lv0(lat) / _unit_lat_lv1())
        cd = int(rem_lon_lv0(lon) / _unit_lon_lv1())
        return str(ab) + str(cd)

    def meshcode_40000(lat, lon):
        e = int(rem_lat_lv1(lat) / _unit_lat_40000())*2 + int(rem_lon_lv1(lon) / _unit_lon_40000()) + 1
        return meshcode_lv1(lat, lon) + str(e)

    def meshcode_20000(lat, lon):
        f = int(rem_lat_40000(lat) / _unit_lat_20000())*2 + int(rem_lon_40000(lon) / _unit_lon_20000()) + 1
        g = 5
        return meshcode_40000(lat, lon) + str(f) + str(g)

    def meshcode_16000(lat, lon):
        e = int(rem_lat_lv1(lat) / _unit_lat_16000())*2
        f = int(rem_lon_lv1(lon) / _unit_lon_16000())*2
        g = 7
        return meshcode_lv1(lat, lon) + str(e) + str(f) + str(g)

    def meshcode_lv2(lat, lon):
        e = int(rem_lat_lv1(lat) / _unit_lat_lv2())
        f = int(rem_lon_lv1(lon) / _unit_lon_lv2())
        return meshcode_lv1(lat, lon) + str(e) + str(f)

    def meshcode_8000(lat, lon):
        e = int(rem_lat_lv1(lat) / _unit_lat_8000())
        f = int(rem_lon_lv1(lon) / _unit_lon_8000())
        g = 6
        return meshcode_lv1(lat, lon) + str(e) + str(f) + str(g)

    def meshcode_5000(lat, lon):
        g = int(rem_lat_lv2(lat) / _unit_lat_5000())*2 + int(rem_lon_lv2(lon) / _unit_lon_5000()) + 1
        return meshcode_lv2(lat, lon) + str(g)

    def meshcode_4000(lat, lon):
        h = int(rem_lat_8000(lat) / _unit_lat_4000())*2 + int(rem_lon_8000(lon) / _unit_lon_4000()) + 1
        i = 7
        return meshcode_8000(lat, lon) + str(h) + str(i)

    def meshcode_2500(lat, lon):
        h = int(rem_lat_5000(lat) / _unit_lat_2500())*2 + int(rem_lon_5000(lon) / _unit_lon_2500()) + 1
        i = 6
        return meshcode_5000(lat, lon) + str(h) + str(i)

    def meshcode_2000(lat, lon):
        g = int(rem_lat_lv2(lat) / _unit_lat_2000())*2
        h = int(rem_lon_lv2(lon) / _unit_lon_2000())*2
        i = 5
        return meshcode_lv2(lat, lon) + str(g) + str(h) + str(i)

    def meshcode_lv3(lat, lon):
        g = int(rem_lat_lv2(lat) / _unit_lat_lv3())
        h = int(rem_lon_lv2(lon) / _unit_lon_lv3())
        return meshcode_lv2(lat, lon) + str(g) + str(h)

    def meshcode_lv4(lat, lon):
        i = int(rem_lat_lv3(lat) / _unit_lat_lv4())*2 + int(rem_lon_lv3(lon) / _unit_lon_lv4()) + 1
        return meshcode_lv3(lat, lon) + str(i)

    def meshcode_lv5(lat, lon):
        j = int(rem_lat_lv4(lat) / _unit_lat_lv5())*2 + int(rem_lon_lv4(lon) / _unit_lon_lv5()) + 1
        return meshcode_lv4(lat, lon) + str(j)

    def meshcode_lv6(lat, lon):
        k = int(rem_lat_lv5(lat) / _unit_lat_lv6())*2 + int(rem_lon_lv5(lon) / _unit_lon_lv6()) + 1
        return meshcode_lv5(lat, lon) + str(k)

    if level == 1:
        return meshcode_lv1(lat, lon)

    if level == 40000:
        return meshcode_40000(lat, lon)

    if level == 20000:
        return meshcode_20000(lat, lon)

    if level == 16000:
        return meshcode_16000(lat, lon)

    if level == 2:
        return meshcode_lv2(lat, lon)

    if level == 8000:
        return meshcode_8000(lat, lon)

    if level == 5000:
        return meshcode_5000(lat, lon)

    if level == 4000:
        return meshcode_4000(lat, lon)

    if level == 2500:
        return meshcode_2500(lat, lon)

    if level == 2000:
        return meshcode_2000(lat, lon)

    if level == 3:
        return meshcode_lv3(lat, lon)

    if level == 4:
        return meshcode_lv4(lat, lon)

    if level == 5:
        return meshcode_lv5(lat, lon)

    if level == 6:
        return meshcode_lv6(lat, lon)

    raise ValueError("the level is unsupported.")