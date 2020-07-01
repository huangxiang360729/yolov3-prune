# yolov3-prune
* 原项目网址：https://github.com/tanluren/yolov3-channel-and-layer-pruning
* 采用的YOLOv3 Pytorch实现：https://github.com/ultralytics/yolov3
* 本项目的将原剪枝代码移植到了新U版的YOLOv3Pytorch实现(2020/06/25)

### 通道剪枝原理
+ 源于：《Learning Efficient Convolutional Networks Through Network Slimming》 (ICCV 2017)
+ 剪枝前的模型中包含了大量的冗余特征（其实就是冗余的通道），我们在剪枝前得先确定哪些通道是不那么重要的（即那些不会对检测任务产生很大影响的通道）。

+ 其实这里就涉及到一个如何评价通道重要性，之前的很多论文提出了不同的方法，包括:
    * Minimum weight（根据卷积核核权重的大小进行修剪，权重越小的被认为越不重要，越应该被剪枝）；
    * Activation（统计激活层的结果大小(其实就是某一卷积层输出的feature map的平均值,当然也可以是累加值)。激活值越小的被认为越不重要，越应该被剪枝）；
    * 信息熵（剪掉某通道会对loss产生的影响，要剪取那些对loss影响小的）；
    * 损失函数泰勒展开的一阶项(从计算的角度看其实就是卷积层输出通道的值乘上损失函数关于该输出通道的梯度并对该乘积求绝对值；从数学意义上理解：采用用通道的有无对loss的影响来表达通道的重要性，即比较有该通道时的loss值和无该通道时的loss值，如果两者相差较大那么说明该通道很重要，但是为了计算的简便性采用两个loss差值的一阶泰勒展开项来代表这个差值)

+ 而这篇论文中采用的原理是：
    * 对BN层的gamma系数进行稀疏化，然后用稀疏化后的gamma系数来评价通道的重要性，因为在<卷积层-BN层-激活函数>中，gamma系数为0就说明了无论卷积层输出的值为何，到了BN层后，它的输出值都变成了：
    * <div align=center><img width="150" height="150" src="https://github.com/huangxiang360729/img-storage/blob/master/yolov3-prune-0.gif"/></div>
    * 其中，gamma即为BN层的gamma系数（在源码中就是bn_module.weight）；beta即为BN层的beta项（在源码中就是bn_module.bias）；X为卷积层的某个输出通道；Y为BN层的在改通道上对应的输出；mu为BN层的均值参数（即bn_module.moving_mean）；sigma为BN层的方差参数（即bn_module.moving_var）；epsilon是为了防止分母为0，可以取1e-16。
    * 上式说明这个卷积层该通道的输出已经对后续模块的前向计算不产影响了，那我们只需要把beta项挪到后续模块中卷积层的bias中或者后续模块的BN层的moving_mean中就可以了：
        + 将beta项作为激活函数的输入可以得到当前模块<卷积层-BN层-激活函数>的输出activation，这个输出要传到下一个模块中，我们可以将它视为剪枝前后模型的计算偏差，为了弥补这个计算偏差，可以这样处理：
            > 1. 当下一个模块是<卷积层-BN层-激活函数>时，BN层的next_bn_module.moving_mean减去Convolution(activation)即可
            > 2. 当下一个模块是<卷积层-激活函数>时，卷积层的next_conv.bias加上Convolution(activation)即可
            > 3. 当下一个模块是跨层连接、pooling、上采样层时，需要把activation传递下去直到遇见卷积模块，然后在卷积模块中进行处理（同1或2）

    * 这样对通道进行剪枝后模型的精度就不会产生变化（如果剪枝的通道对应的gamma系数为0），等于稀疏后模型的精度，而稀疏后模型的精度几乎等于原始模型的精度

    * 但是稀疏化只能把大部分gamma系数驱赶到接近0的值，而不是真正的0，所以剪枝后的模型还是会有精度的下降(当然剪枝后模型精度甚至有可能超越原始模型的精度)，这时通过微调就可以把精度恢复（前提是剪枝率没有设的太高，如果剪枝率设的过高，会把一些gamma系数不是很接近0的通道也给剪掉，这时精度就会掉很多）

    * 如果不进行稀疏化训练就直接按照gamma系数进行从小到大的排序，然后减去top-N，那么剪枝后模型的精度肯定非常低（接近0，或者0），通过微调或许可以恢复模型精度，但是这就相当于从零训练一个神经网络，而且如果剪枝率设的很大的化，很有可能微调后模型精度降很多。

### 剪枝步骤
1. 正常训练得到模型的参数（也可以采用迁移学习的方式获取模型的参数）
2. 稀疏化训练
    * 这里说的稀疏化训练指的是在损失函数中添加关于BN层的gamma系数的L1正则化项，然后反向传递的时候gamma系数会相应的进行梯度更新（实际上代码里并没有对损失函数进行修改，而是对那些能够被剪枝的通道对应的gamma系数的梯度添加上L1正则化惩罚项，然后在反向传播时，gamma系数会减掉lr乘上梯度。值得注意的是gammma系数的梯度包含了损失函数对其求导项也包含了L1正则化惩罚项）
    * gamma系数的梯度更新会使得大量gamma系数的值趋于0，论文认为那些趋于0的γ系数对应的通道都是不重要的，可以剪枝减掉
3. 剪枝
    * 需要对ResNet和DenseNet中的跨层连接特别考虑
4. 微调
    * 为了恢复模型的精度
5. Multi-pass Scheme（迭代剪枝）
    * 返回第2步进行新一轮的剪枝

### 稀疏方式
+ 这里说的稀疏方式指的是对哪些层的gamma系数能够进行稀疏化，哪些层的gamma系数不能够稀疏化
+ 总共有两种稀疏方式：：
    * 稀疏方式0，在运行代码时需要指定prune = 0: 
        + 稀疏化针对的是CBL（ Conv-Bn-Leaky_relu，包含BN层的卷积模块），而不包含BN层的卷积模块（yolo层{检测器的head,用于bbox分类和回归}的前一个卷积模块不包含BN层）是不能被稀疏化的
        + 这些CBL中，SPP结构前的一个CBL不剪,上采样层前的CBL也不剪，这是模型结构的限制
        + 生成shortcut的两个输入特征图的CBL不剪，所以不用稀疏化，这是稀疏方式0决定的
    * 稀疏方式1，在运行代码时需要指定prune = 1: 
        + 稀疏化针对的是CBL，而不包含BN层的卷积模块（yolo层的前一个卷积模块不包含BN层）是不能被稀疏化的
        + 这些CBL中，SPP结构前的一个CBL不剪,上采样层前的CBL也不剪，这是模型结构的限制
        + 生成shortcut的两个输入特征图的CBL有可能需要剪枝，所以也需要稀疏化，这是稀疏方式1决定的
+ 剪枝方式依赖于稀疏方式
    * 朴素的剪枝方式（不对shortcut直连的前一个CBL和跨层连接CBL的进行剪枝）采用稀疏方式0，而其他剪枝方式采用稀疏方式1

### 剪枝方式
+ 朴素的剪枝方式（prune.py）
    * 不对shortcut直连的前一个CBL和跨层连接CBL的进行剪枝

***
+ shortcut_prune.py
    * 针对shortcut,前面的shortcut跨层连接的CBL剪完后，后面对应的该shotcut直连的前一个CBL也要采用同样的filter掩码进行剪枝（为了满足shortcut的对应通道相加）
    * 剪枝步骤
        + CBL_idx，包含BN层的卷积模块构成的索引列表
        + ignore_idx，模型结构的限制下，不能够被剪枝的CBL构成的索引列表（spp前一个CBL不剪,上采样层前的CBL不剪)
        + prune_idx，能够被剪枝的CBL构成的索引列表（CBL_idx - ignore_idx）
        + sort_prune_idx，从prune_idx中去掉shortcut直连的前一个CBL，注意这里的sort_prune_idx依然包含shortcut跨层连接的CBL
        + sort_bn，对sort_prune_idx包含的所有CBL的gamma系数进行排序
        + threshold，根据sort_bn和剪枝率求得gamma系数阈值，sort_bn中所有小于该threshold的gamma系数置0，而大于或等于该threshold的gamma系数不变
        + filters_mask
            * 表示模型中每一个CBL对应的剪枝掩码
            * 对gamma系数小于threshold的filter设置剪枝掩码为0，表示剪枝；对gamma系数大于或等于threshold的filter设置剪枝掩码为1，表示不剪枝
            * 需要特别考虑的地方：shortcut直连的前一个CBL的filter掩码应该设置为该shortcut跨层连接的CBL的filter掩码（即前面的shortcut跨层连接的CBL剪完后，后面对应的该shotcut直连的前一个CBL也要采用同样的filter掩码进行剪枝（为了满足shortcut的对应通道相加））
            * 不能够被剪枝的CBL的剪枝掩码设置全1
        + 根据filter_mask生成剪枝后的紧凑模型，并从剪枝前的宽松模型中加载参数
        
***
+ slim_prune.py
    * 针对shortcut,将shortcut相关联的所有CBL的filter掩码对0求交集，即只有所有CBL在某个位置的filter被剪枝，则所有CBL在该位置的filter才能被剪枝， 但凡有一个CBL中的在该位置的filter不被剪枝，则所有CBL在该位置的filter都不被剪枝
    * 剪枝步骤
        + CBL_idx，包含BN层的卷积模块构成的索引列表
        + ignore_idx，模型结构的限制下，不能够被剪枝的CBL构成的索引列表（spp前一个CBL不剪,上采样层前的CBL不剪)
        + prune_idx，能够被剪枝的CBL构成的索引列表（CBL_idx - ignore_idx）
        + sort_bn，对prune_idx包含的所有CBL的gamma系数进行排序
        + threshold，根据sort_bn和剪枝率求得gamma系数阈值，sort_bn中所有小于该threshold的gamma系数置0，而大于或等于该threshold的gamma系数不变
        + filters_mask
            * 表示模型中每一个CBL对应的剪枝掩码
            * 对gamma系数置0的filter设置剪枝掩码为0，表示剪枝；对gamma系数置1的filter设置剪枝掩码为1，表示不剪枝
            * 需要特别考虑的地方：将shortcut相关联的所有CBL的filter掩码对0求交集，即只有所有CBL在某个位置的filter被剪枝，则所有CBL在该位置的filter才能被剪枝， 但凡有一个CBL中的在该位置的filter不被剪枝，则所有CBL在该位置的filter都不被剪枝
                * shortcut相关联的所有CBL指的是shortcut直连的前一个CBL和shortcut跨层连接相关的CBL，shortcut的跨层连接可能是一个CBL也可能是一个shortcut，当shortcut的跨层连接是shortcut时就存在一个级联的多个CBL(3个及3个以上)共享同一个filter掩码
            * 不能够被剪枝的CBL的剪枝掩码设置全1
        + 根据filter_mask生成剪枝后的紧凑模型，并从剪枝前的宽松模型中加载参数

***
+ 层剪枝（layer_prune.py）
    * 这里的层剪枝实际上是针对shortcut的剪枝，针对每一个shortcut的前一个CBL的gamma均值进行排序，取最小的若干个shortcut进行剪枝。为保证yolov3结构完整，这里每剪一个shortcut模块，会同时剪掉一个shortcut模块和它前面的两个卷积模块（Residual Block）。并且只考虑剪主干网络（backbone）中的shortcut模块。
    * Residual Block是不会改变通道数目和特征图大小的，层剪枝正是利用了这一点，关于评价某一层的重要性，这里采用的是shortcut的前一个CBL的gamma均值，这个也是和剪枝论文的原理相通的，即稀疏后gamma系数越小的越不重要，而一个Residual Block需要学习的特征正是残差F(x)=H(x)-x，如果shortcut的前一个CBL的gamma均值（实际上是gamma系数绝对值的均值）较小，那么F(x)的输出值的绝对值也会较小，也即该Residual Block没有存在的必要。
    
***
+ 同时通道剪枝和层剪枝（layer_channel_prune.py）
    * 没有新的东西，实际上是先进行通道剪枝，然后进行层剪枝，只不过把两个python脚本（slim_prune.py和layer_prune.py）的主要功能写在一块了
