# @Author: Lu Shaohao(Bravo)
# @Date:   2018-11-13 10:33:24
# @Last Modified by:   Lu Shaohao(Bravo)
# @Last Modified time: 2019-02-27 16:30:28
CMD="2019-2-27 update"
#!/bin/bash
echo "start git push"
git add .
git commit -m $CMD
git push 
echo "git push success!"