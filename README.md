下面这个版本我已经按照你上传的 **model.py + main.py** 整理成了真正能拿着讲的版本。

注意：

* model.py 行号我能基本确定；
* main.py 由于 file_search 没给完整行号，我只能给出**大概区间**（误差 ±10 行左右）；
* 你答辩前打开 VSCode 看一下即可。

---

# FA-Neg-CG-CAM 代码讲解（3分钟）

## 一、整体思路（15秒）

今天主要介绍我们在原始 FA（Forced Prompt Learning）基础上的三个改进：

```text
1. Negative Prompt
2. Correct Gate
3. Class-Adaptive Margin
```

整体流程如下：

```text
Image
   ↓
CLIP Image Encoder
   ↓
Image Features
   ↓
Positive Prompt
Negative Prompt
   ↓
Similarity Score
   ↓
FA Loss
Negative Loss
Margin Loss
   ↓
Total Loss
```

我们没有修改 CLIP 主体结构，只是在 Prompt 和 Loss 上进行了增强。

---

# 二、Negative Prompt（40秒）

## 文件

```text
model.py
```

## 位置

大约：

```text
Line 210~225
```

搜索：

```python
self.negative_prompt_learner
```

---

## 展示代码

```python
self.negative_prompt_learner = PromptLearner(
    cfg,
    classnames,
    clip_model,
    cfg['negative_template'],
    ...
)
```

---

## 讲解

原始 FA 只有 Positive Prompt。

例如：

```text
a photo of a forest
```

模型学习的是：

```text
这是什么
```

我们额外增加了一组 Negative Prompt。

例如：

```text
not forest
not river
not sea
```

对应代码中的：

```python
self.negative_prompt_learner
```

这样模型不仅学习：

```text
这是什么
```

同时学习：

```text
这不是什么
```

从而增强类别判别能力。

---

# 三、Negative Similarity（30秒）

## 文件

```text
model.py
```

## 位置

大约：

```text
Line 280~300
```

搜索：

```python
logits_neg
```

---

## 展示代码

```python
negative_text_features =
    self.text_encoder(
        negative_prompts,
        negative_tokenized_prompts
    )
```

以及：

```python
logits_neg =
    logit_scale *
    image_features
    @
    negative_text_features.T
```

---

## 讲解

这里利用 CLIP Text Encoder 提取：

```text
Negative Prompt Feature
```

然后与：

```text
Image Feature
```

计算余弦相似度。

得到：

```python
logits_neg
```

表示：

```text
图像有多像
“not class”
```

例如：

```text
Forest图片

如果:
not forest 分数很高

说明模型混淆了
```

因此后面需要进一步约束。

---

# 四、Negative Loss（40秒）

## 文件

```text
main.py
```

## 位置

大约：

```text
Line 420~450
```

搜索：

```python
neg_targets
```

---

## 展示代码

```python
neg_targets =
    build_negative_targets(
        target,
        class_num
    )
```

以及：

```python
loss_neg_per_sample =
F.binary_cross_entropy_with_logits(
    logits_neg_for_loss,
    neg_targets.float(),
    reduction='none'
).mean(dim=1)
```

---

## 讲解

这里构造 Negative Label。

假设：

```text
真实类别 Forest
```

则：

```text
not Forest = 0

not River = 1
not Sea = 1
...
```

然后利用 BCE Loss：

```python
binary_cross_entropy_with_logits
```

计算：

```text
Negative Loss
```

其作用是：

```text
让 Forest 图片
远离 not Forest
```

---

# 五、Correct Gate（50秒）

## 文件

```text
main.py
```

## 位置

大约：

```text
Line 470~500
```

搜索：

```python
prob_true
```

或者：

```python
gate =
```

---

## 展示代码

```python
prob_true =
F.softmax(
    just_forced_logits.detach()
    /
    cfg['neg_loss_scale'],
    dim=1
).gather(...)
```

然后：

```python
gate =
cfg['conf_gate_min']
+
(1-cfg['conf_gate_min'])
*
prob_true.pow(
    cfg['conf_gate_gamma']
)
```

---

## 讲解

这里计算：

```python
prob_true
```

表示：

```text
模型认为当前样本属于真实类别的概率
```

例如：

```text
Forest 0.95
River 0.03
Sea 0.02
```

则：

```text
prob_true = 0.95
```

说明模型已经非常确信。

随后构造：

```python
gate
```

其作用是：

```text
容易样本
gate大

困难样本
gate小
```

这样：

```text
先学分类

再加强负向约束
```

避免训练初期过度干扰。

---

# 六、Class-Adaptive Margin（50秒）

## 文件

```text
main.py
```

## 第一处

大约：

```text
Line 70~140
```

搜索：

```python
compute_class_adaptive_margins
```

---

## 展示代码

```python
prototypes =
feat_sum /
counts

sim_matrix =
prototypes
@
prototypes.t()
```

---

## 讲解

这里利用：

```text
CLIP Frozen Image Feature
```

构造：

```text
Class Prototype
```

然后计算：

```text
类别与类别之间的相似度
```

例如：

```text
Forest
Pasture
```

非常接近。

则：

```text
Margin更大
```

反之：

```text
Margin更小
```

---

## 第二处

大约：

```text
Line 450~470
```

搜索：

```python
margin_y
```

---

## 展示代码

```python
margin_y =
class_margins.gather(
    0,
    target
)
```

然后：

```python
loss_margin_per_sample =
F.relu(
    margin_y
    +
    neg_true
    -
    pos_true
)
```

---

## 讲解

这里实现：

```text
Class Adaptive Margin
```

约束：

```text
positive score

必须明显高于

negative score
```

即：

```text
positive_score(y)

>

negative_score(not y)
```

从而增强类别边界。

---

# 七、最终损失（20秒）

## 文件

```text
main.py
```

## 位置

大约：

```text
Line 500
```

搜索：

```python
loss =
```

---

## 展示代码

```python
loss =
loss_global
+
cfg['lambda_neg']
*
loss_neg
+
cfg['lambda_margin']
*
loss_margin
```

---

## 讲解

最终损失由三部分组成：

```text
FA Loss

+
Negative Loss

+
Margin Loss
```

其中：

```text
FA Loss
保证分类能力

Negative Loss
提供负向语义监督

Correct Gate
动态控制监督强度

CAM
增强类别边界
```

---

# 结束总结（10秒）

```text
我们保持了原始 FA 的整体结构不变，

新增了 Negative Prompt、
Correct Gate 和
Class-Adaptive Margin。

其中：

Negative Prompt 负责学习“这不是什么”，

Correct Gate 负责动态调节监督强度，

Class-Adaptive Margin 负责增强类别边界，

最终提升了 OOD 检测性能。
```

这个版本正常语速基本 **2分50秒~3分10秒**，已经非常接近课程答辩的最佳长度了。
下面这个版本我已经按照你上传的 **model.py + main.py** 整理成了真正能拿着讲的版本。

注意：

* model.py 行号我能基本确定；
* main.py 由于 file_search 没给完整行号，我只能给出**大概区间**（误差 ±10 行左右）；
* 你答辩前打开 VSCode 看一下即可。

---

# FA-Neg-CG-CAM 代码讲解（3分钟）

## 一、整体思路（15秒）

今天主要介绍我们在原始 FA（Forced Prompt Learning）基础上的三个改进：

```text
1. Negative Prompt
2. Correct Gate
3. Class-Adaptive Margin
```

整体流程如下：

```text
Image
   ↓
CLIP Image Encoder
   ↓
Image Features
   ↓
Positive Prompt
Negative Prompt
   ↓
Similarity Score
   ↓
FA Loss
Negative Loss
Margin Loss
   ↓
Total Loss
```

我们没有修改 CLIP 主体结构，只是在 Prompt 和 Loss 上进行了增强。

---

# 二、Negative Prompt（40秒）

## 文件

```text
model.py
```

## 位置

大约：

```text
Line 210~225
```

搜索：

```python
self.negative_prompt_learner
```

---

## 展示代码

```python
self.negative_prompt_learner = PromptLearner(
    cfg,
    classnames,
    clip_model,
    cfg['negative_template'],
    ...
)
```

---

## 讲解

原始 FA 只有 Positive Prompt。

例如：

```text
a photo of a forest
```

模型学习的是：

```text
这是什么
```

我们额外增加了一组 Negative Prompt。

例如：

```text
not forest
not river
not sea
```

对应代码中的：

```python
self.negative_prompt_learner
```

这样模型不仅学习：

```text
这是什么
```

同时学习：

```text
这不是什么
```

从而增强类别判别能力。

---

# 三、Negative Similarity（30秒）

## 文件

```text
model.py
```

## 位置

大约：

```text
Line 280~300
```

搜索：

```python
logits_neg
```

---

## 展示代码

```python
negative_text_features =
    self.text_encoder(
        negative_prompts,
        negative_tokenized_prompts
    )
```

以及：

```python
logits_neg =
    logit_scale *
    image_features
    @
    negative_text_features.T
```

---

## 讲解

这里利用 CLIP Text Encoder 提取：

```text
Negative Prompt Feature
```

然后与：

```text
Image Feature
```

计算余弦相似度。

得到：

```python
logits_neg
```

表示：

```text
图像有多像
“not class”
```

例如：

```text
Forest图片

如果:
not forest 分数很高

说明模型混淆了
```

因此后面需要进一步约束。

---

# 四、Negative Loss（40秒）

## 文件

```text
main.py
```

## 位置

大约：

```text
Line 420~450
```

搜索：

```python
neg_targets
```

---

## 展示代码

```python
neg_targets =
    build_negative_targets(
        target,
        class_num
    )
```

以及：

```python
loss_neg_per_sample =
F.binary_cross_entropy_with_logits(
    logits_neg_for_loss,
    neg_targets.float(),
    reduction='none'
).mean(dim=1)
```

---

## 讲解

这里构造 Negative Label。

假设：

```text
真实类别 Forest
```

则：

```text
not Forest = 0

not River = 1
not Sea = 1
...
```

然后利用 BCE Loss：

```python
binary_cross_entropy_with_logits
```

计算：

```text
Negative Loss
```

其作用是：

```text
让 Forest 图片
远离 not Forest
```

---

# 五、Correct Gate（50秒）

## 文件

```text
main.py
```

## 位置

大约：

```text
Line 470~500
```

搜索：

```python
prob_true
```

或者：

```python
gate =
```

---

## 展示代码

```python
prob_true =
F.softmax(
    just_forced_logits.detach()
    /
    cfg['neg_loss_scale'],
    dim=1
).gather(...)
```

然后：

```python
gate =
cfg['conf_gate_min']
+
(1-cfg['conf_gate_min'])
*
prob_true.pow(
    cfg['conf_gate_gamma']
)
```

---

## 讲解

这里计算：

```python
prob_true
```

表示：

```text
模型认为当前样本属于真实类别的概率
```

例如：

```text
Forest 0.95
River 0.03
Sea 0.02
```

则：

```text
prob_true = 0.95
```

说明模型已经非常确信。

随后构造：

```python
gate
```

其作用是：

```text
容易样本
gate大

困难样本
gate小
```

这样：

```text
先学分类

再加强负向约束
```

避免训练初期过度干扰。

---

# 六、Class-Adaptive Margin（50秒）

## 文件

```text
main.py
```

## 第一处

大约：

```text
Line 70~140
```

搜索：

```python
compute_class_adaptive_margins
```

---

## 展示代码

```python
prototypes =
feat_sum /
counts

sim_matrix =
prototypes
@
prototypes.t()
```

---

## 讲解

这里利用：

```text
CLIP Frozen Image Feature
```

构造：

```text
Class Prototype
```

然后计算：

```text
类别与类别之间的相似度
```

例如：

```text
Forest
Pasture
```

非常接近。

则：

```text
Margin更大
```

反之：

```text
Margin更小
```

---

## 第二处

大约：

```text
Line 450~470
```

搜索：

```python
margin_y
```

---

## 展示代码

```python
margin_y =
class_margins.gather(
    0,
    target
)
```

然后：

```python
loss_margin_per_sample =
F.relu(
    margin_y
    +
    neg_true
    -
    pos_true
)
```

---

## 讲解

这里实现：

```text
Class Adaptive Margin
```

约束：

```text
positive score

必须明显高于

negative score
```

即：

```text
positive_score(y)

>

negative_score(not y)
```

从而增强类别边界。

---

# 七、最终损失（20秒）

## 文件

```text
main.py
```

## 位置

大约：

```text
Line 500
```

搜索：

```python
loss =
```

---

## 展示代码

```python
loss =
loss_global
+
cfg['lambda_neg']
*
loss_neg
+
cfg['lambda_margin']
*
loss_margin
```

---

## 讲解

最终损失由三部分组成：

```text
FA Loss

+
Negative Loss

+
Margin Loss
```

其中：

```text
FA Loss
保证分类能力

Negative Loss
提供负向语义监督

Correct Gate
动态控制监督强度

CAM
增强类别边界
```

---

# 结束总结（10秒）

```text
我们保持了原始 FA 的整体结构不变，

新增了 Negative Prompt、
Correct Gate 和
Class-Adaptive Margin。

其中：

Negative Prompt 负责学习“这不是什么”，

Correct Gate 负责动态调节监督强度，

Class-Adaptive Margin 负责增强类别边界，

最终提升了 OOD 检测性能。
```

这个版本正常语速基本 **2分50秒~3分10秒**，已经非常接近课程答辩的最佳长度了。
下面这个版本我已经按照你上传的 **model.py + main.py** 整理成了真正能拿着讲的版本。

注意：

* model.py 行号我能基本确定；
* main.py 由于 file_search 没给完整行号，我只能给出**大概区间**（误差 ±10 行左右）；
* 你答辩前打开 VSCode 看一下即可。

---

# FA-Neg-CG-CAM 代码讲解（3分钟）

## 一、整体思路（15秒）

今天主要介绍我们在原始 FA（Forced Prompt Learning）基础上的三个改进：

```text
1. Negative Prompt
2. Correct Gate
3. Class-Adaptive Margin
```

整体流程如下：

```text
Image
   ↓
CLIP Image Encoder
   ↓
Image Features
   ↓
Positive Prompt
Negative Prompt
   ↓
Similarity Score
   ↓
FA Loss
Negative Loss
Margin Loss
   ↓
Total Loss
```

我们没有修改 CLIP 主体结构，只是在 Prompt 和 Loss 上进行了增强。

---

# 二、Negative Prompt（40秒）

## 文件

```text
model.py
```

## 位置

大约：

```text
Line 210~225
```

搜索：

```python
self.negative_prompt_learner
```

---

## 展示代码

```python
self.negative_prompt_learner = PromptLearner(
    cfg,
    classnames,
    clip_model,
    cfg['negative_template'],
    ...
)
```

---

## 讲解

原始 FA 只有 Positive Prompt。

例如：

```text
a photo of a forest
```

模型学习的是：

```text
这是什么
```

我们额外增加了一组 Negative Prompt。

例如：

```text
not forest
not river
not sea
```

对应代码中的：

```python
self.negative_prompt_learner
```

这样模型不仅学习：

```text
这是什么
```

同时学习：

```text
这不是什么
```

从而增强类别判别能力。

---

# 三、Negative Similarity（30秒）

## 文件

```text
model.py
```

## 位置

大约：

```text
Line 280~300
```

搜索：

```python
logits_neg
```

---

## 展示代码

```python
negative_text_features =
    self.text_encoder(
        negative_prompts,
        negative_tokenized_prompts
    )
```

以及：

```python
logits_neg =
    logit_scale *
    image_features
    @
    negative_text_features.T
```

---

## 讲解

这里利用 CLIP Text Encoder 提取：

```text
Negative Prompt Feature
```

然后与：

```text
Image Feature
```

计算余弦相似度。

得到：

```python
logits_neg
```

表示：

```text
图像有多像
“not class”
```

例如：

```text
Forest图片

如果:
not forest 分数很高

说明模型混淆了
```

因此后面需要进一步约束。

---

# 四、Negative Loss（40秒）

## 文件

```text
main.py
```

## 位置

大约：

```text
Line 420~450
```

搜索：

```python
neg_targets
```

---

## 展示代码

```python
neg_targets =
    build_negative_targets(
        target,
        class_num
    )
```

以及：

```python
loss_neg_per_sample =
F.binary_cross_entropy_with_logits(
    logits_neg_for_loss,
    neg_targets.float(),
    reduction='none'
).mean(dim=1)
```

---

## 讲解

这里构造 Negative Label。

假设：

```text
真实类别 Forest
```

则：

```text
not Forest = 0

not River = 1
not Sea = 1
...
```

然后利用 BCE Loss：

```python
binary_cross_entropy_with_logits
```

计算：

```text
Negative Loss
```

其作用是：

```text
让 Forest 图片
远离 not Forest
```

---

# 五、Correct Gate（50秒）

## 文件

```text
main.py
```

## 位置

大约：

```text
Line 470~500
```

搜索：

```python
prob_true
```

或者：

```python
gate =
```

---

## 展示代码

```python
prob_true =
F.softmax(
    just_forced_logits.detach()
    /
    cfg['neg_loss_scale'],
    dim=1
).gather(...)
```

然后：

```python
gate =
cfg['conf_gate_min']
+
(1-cfg['conf_gate_min'])
*
prob_true.pow(
    cfg['conf_gate_gamma']
)
```

---

## 讲解

这里计算：

```python
prob_true
```

表示：

```text
模型认为当前样本属于真实类别的概率
```

例如：

```text
Forest 0.95
River 0.03
Sea 0.02
```

则：

```text
prob_true = 0.95
```

说明模型已经非常确信。

随后构造：

```python
gate
```

其作用是：

```text
容易样本
gate大

困难样本
gate小
```

这样：

```text
先学分类

再加强负向约束
```

避免训练初期过度干扰。

---

# 六、Class-Adaptive Margin（50秒）

## 文件

```text
main.py
```

## 第一处

大约：

```text
Line 70~140
```

搜索：

```python
compute_class_adaptive_margins
```

---

## 展示代码

```python
prototypes =
feat_sum /
counts

sim_matrix =
prototypes
@
prototypes.t()
```

---

## 讲解

这里利用：

```text
CLIP Frozen Image Feature
```

构造：

```text
Class Prototype
```

然后计算：

```text
类别与类别之间的相似度
```

例如：

```text
Forest
Pasture
```

非常接近。

则：

```text
Margin更大
```

反之：

```text
Margin更小
```

---

## 第二处

大约：

```text
Line 450~470
```

搜索：

```python
margin_y
```

---

## 展示代码

```python
margin_y =
class_margins.gather(
    0,
    target
)
```

然后：

```python
loss_margin_per_sample =
F.relu(
    margin_y
    +
    neg_true
    -
    pos_true
)
```

---

## 讲解

这里实现：

```text
Class Adaptive Margin
```

约束：

```text
positive score

必须明显高于

negative score
```

即：

```text
positive_score(y)

>

negative_score(not y)
```

从而增强类别边界。

---

# 七、最终损失（20秒）

## 文件

```text
main.py
```

## 位置

大约：

```text
Line 500
```

搜索：

```python
loss =
```

---

## 展示代码

```python
loss =
loss_global
+
cfg['lambda_neg']
*
loss_neg
+
cfg['lambda_margin']
*
loss_margin
```

---

## 讲解

最终损失由三部分组成：

```text
FA Loss

+
Negative Loss

+
Margin Loss
```

其中：

```text
FA Loss
保证分类能力

Negative Loss
提供负向语义监督

Correct Gate
动态控制监督强度

CAM
增强类别边界
```

---

# 结束总结（10秒）

```text
我们保持了原始 FA 的整体结构不变，

新增了 Negative Prompt、
Correct Gate 和
Class-Adaptive Margin。

其中：

Negative Prompt 负责学习“这不是什么”，

Correct Gate 负责动态调节监督强度，

Class-Adaptive Margin 负责增强类别边界，

最终提升了 OOD 检测性能。
```

这个版本正常语速基本 **2分50秒~3分10秒**，已经非常接近课程答辩的最佳长度了。
