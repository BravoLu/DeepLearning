# -*- coding: utf-8 -*-
# @Author: Lu Shaohao(Bravo)
# @Date:   2019-12-04 14:38:56
# @Last Modified by:   Lu Shaohao(Bravo)
# @Last Modified time: 2019-12-04 14:40:57
import re 

strings = '<div class="nam">中国</div>'

res = re.findall('<div class=".*">(.*?)</div>', strings)
print(res)