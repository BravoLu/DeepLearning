# PCB-RPP net
了解PCB-RPP网络结构并验证PCB-RPP相关性能是否与论文一致，然后记录一些遇到的issue。
下图是论文中列出的一些不同结构网络的性能。
<br>
![img](image/Benchmark.png)
<br>
[源码地址](https://github.com/syfafterzy/PCB_RPP_for_reID)
<br> 
[论文地址](https://arxiv.org/pdf/1711.09349.pdf)
## dataset路径问题
linux下换行为/n，windows下为/r/n，secureCRT远程控制server时键入的enter貌似为/r/n，所以最好不要将路径参数放在命令最后。
## PCB
### Structure
![img](image/structure.png)


### some issue
如果在我的server直接运行源码会出现几处issue
* issue 1:
```
torch.autograd.backward([ loss0, loss1, loss2, loss3, loss4, loss5],[torch.ones(1).cuda(), torch.ones(1).cuda(), torch.ones(1).cuda(),torch.ones(1).cuda(),torch.ones(1).cuda(),torch.ones(1).cuda(),torch.ones(1).cuda()]) 
-----------------------------------------------------------------------------------------------------------
torch.autograd.backward([ loss0, loss1, loss2, loss3, loss4, loss5]) 
--------------------------------------------------------------------
查看官方文档：由于loss0~loss5是标量，所以是不需要variable_grad这个参数的，该参数在Loss为多维时相当于为每个维度设置不同的学习率。
```
* issue 2:
由于原文作业用的为Python 2， 所以有些/要换成//


