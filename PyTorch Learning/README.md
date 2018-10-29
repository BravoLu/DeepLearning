### 变量
```
    variable = Variable(tensor,requires_grad=True) 
    #requires_grad表示是否参与误差反向传播，是否计算梯度

    variable #Variable形式
    Variable是Pytorch中autograd自动微分模块的核心。封装了Tensor。主要包含3个属性
    1.data  保存Variable所包含的Tensor
    2.grad  保存data对应的梯度,grad也是一个Variable,而不是一个Tensor,和data的形状一样
    3. grad_fn: 指向一个Function对象，这个Function用来反向传播计算输入的梯度。
    
    variable.data #tensor形式
    variable.data.numpy()  #numpy形式

    import torch.nn.functional as F  #激励函数
    F.relu(tensor).data

    torch.max(F.softmax(out),1)[1]  
  
    #快速搭建法
    net = torch.nn.Sequential(
        torch.nn.Linear(1,10),
        torch.nn.ReLu(),
        torch.nn.Linear()
    )

    #保存
    torch.manual_seed(1)
    #保存整个网络
    torch.save(net1,'net.pkl')
    #只保存网络中的参数

    torch.save(net1.state_dict(),'net_params.pkl')
    def restore_net():
        net2 = torch.load('net.pkl')
        prediction = net2(x)

    #优化方法
    SGD
    Momentum
    AdaGrad
    RMSProp
    Adam
    ReLU: Rectified Linear Units layer,ReLU layer
    filter(滤波器)和Kernel(卷积核)
```
### Pytorch torchvisio transform
```
    1. torchvision.transforms.Compose(transforms)
    参数：
    transforms: 由transform构成的列表

    2.对PIL.Image进行变换
    torchvision.transforms.Scale(size, interpolation=2)
    参数：size为最小边长，将最小边长改为size

    torchvision.transforms.CenterCrop(size)
    将给定的PIL.Image进行中心切割，得到给定的size，size可以是tuple,(target_height,target_width),size也可以是一个Integer，切出来为正方形

    transform.RandomCrop
    切割中心点的位置随机选取，size可以是tuple也可以是Integer

    transforms.RandomHorizontalFlip
    随机水平翻转给定的PIL.Image,概率为0.5。

    transforms.RandomSizedCrop(size,interpolation=2)
    先将给定的PIL.Image随机切，然后再resize成给定的size大小

    transforms.Pad(padding,fill=0)
    将给定的PIL.Image的所有边用给定的pad value填充。padding:要填充多少像素 fill:用什么值填充 

```

### 对Tensor进行变换
```
    Normalize(mean,std)
    给定均值：(R,G,B) 方差 (R,G,B), 将会把Tensor正则化。即:Normalized_image=(image-mean)/std

    ToTensor
    把一个取值范围是[0,255]的PIL.Image或者shape为(H,W,C)的numpy.ndarray转换成形状为[C,H,W],取值范围是[0,1,0]的torch.FloatTensor

    ToPILImage
    将shape为(C,H,W)的Tensor或shape为(H,W,C)的numpy.ndarray 转换为PIL.Image,值不变。

    通用变换
    Lambda
```

### torchvision.utils
```
    torchvision.utils
    torchvision.utils.make_grid(tensor,nrow=8,padding=2,normalize=False,range=None,scale_each=False )
    e.g. see Make_grid.py
```

### Autograd mechanics
```
    requires_grad:当你想要冻结部分模型时，或者你事先知道不会使用某些参数的梯度。例如，如果要对预先训练的CNN进行优化，只要切换冻结模型中的requires_grad标志就足够了，知道计算到最后一层才会保存中间缓冲区，其中的仿射变换将使用需要梯度的权重并且网络的输出也将需要他们

    e.g. 
    model = torchvision.models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    #Replace the last fully-connected layer
    #Parameters of newly constructed modules have requires_grad= True by default

    model.fc = nn.linear(512,100)
    optimizer = optim.SGD(model.fc.parameters(),lr=1e-2, momentum=0.9)
```

### torch
```
    torch.numel(input) -> int 
    返回input张量中的元素个数

    torch.set_printoptions(precision=None,threshold=None,edgeitems=None,linewidth=None,profile=None)
    设置打印选项
    precision - 浮点数输出精度位数（默认为8）
    threshold - 阈值，触发汇总显示而不是完全显示的数组元素总数
    edgeitems - 汇总显示中，每维(轴)两端显示的项数（默认值为3）
    linewidth - 用于插入行间隔的每行字符数（默认为80）。
    profile  - pretty打印的完全默认值

    torch.eye(n,m=None,out=None)
    返回一个2维张量，对角线位置全1，其他位置全0，m为列数

    torch.from_numpy(ndarray) -> Tensor
    将numpy.ndarray 转换为pytorch的Tensor。返回的张量tensor和numpy的ndarray共享一个内存空间。修改一个会导致另外一个也被修改。返回的张量不能改变大小。

    torch.linspace(start,end,steps=100,out=None) -> Tensor
    返回一个1维张量，包含在区间start和end上均匀间隔的steps个点。输出一维张量的长度为steps

    torch.logspace(start,end,steps=100,out=None) -> Tensor

    torch.ones(*sizes,out=None) -> Tensor
    返回一个全为1的张量，形状为sizes定义

    torch.rand(*size,out=None) -> Tensor 
    返回包含从区间[0,1)的均匀分布中抽取的一组随机数。

    torch.randn(*sizes,out=None) -> Tensor
    返回一个张量，包含从标准正太分布（均值为0，方差为1，即高斯白噪声）中抽取一组随机数，形状有可变参数sizes定义。

    torch.randperm(n,out=None) -> LongTensor
    返回一个从0到n-1的随机整数排列

    torch.arange(start,end,step=1,out=None) -> Tensor
    torch.range(start,end,step=1,out=None) -> Tensor 元素比arange多一个

    torch.zeros(*size,out=None)

    torch.cat(inputs,dimension=0) -> Tensor
    dimension = 0 增加行
    dimension = 1 增加列

    torch.chunk(tensor,chunks,dim=0)  在给定维度(轴)上对张量分块

    torch.gather(input,dim,index,out=None) 
    沿给定轴dim,将输入索引张量index指定位置的值进行聚合

    torch.index_select(input,dim,index,out=None) -> Tensor
    沿着指定维度对输入进行切片，取index中指定的相应项(index为一个LongTensor)，然后返回到一个新的张量， 返回的张量与原始张量_Tensor_有相同的维度(在指定轴上)。

    torch.nonzero(input, out=None) -> LongTensor
    返回一个包含输入input中非0元素索引的张量，输出张量中的每行包含输入中非零元素的索引





```