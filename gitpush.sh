# @Author: Lu Shaohao(Bravo)
# @Date:   2018-11-13 10:33:24
# @Last Modified by:   Lu Shaohao(Bravo)
# @Last Modified time: 2018-11-13 10:34:28

#!/bin/bash
echo "start git push"
git add .
git commit -m $1
echo $1
git push 
echo "git push success!"