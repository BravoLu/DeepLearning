# -*- coding: utf-8 -*-
# @Author: Lu Shaohao(Bravo)
# @Date:   2018-10-29 19:59:32
# @Last Modified by:   Lu Shaohao(Bravo)
# @Last Modified time: 2018-10-29 20:02:59


#批量生成pyc文件
#针对一个目录下所有的py文件进行编译，python提供了一个模块compileall


import compileall


compileall.compile_dir('../Python Learning')