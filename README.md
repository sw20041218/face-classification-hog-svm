# 19×19 人脸分类：HOG 与 RBF-SVM 六轮实验

本仓库整理模式识别课程中人脸 / 非人脸二分类任务的六轮实验，包含可运行源码、课程提供的训练数据与无标签测试数据、保存的模型以及原始实验记录。六轮均采用对比度归一化像素与自实现 HOG 联合特征，使用单个 RBF 核 SVM；第二轮增加四方向单像素平移增强；第三轮比较局部对比度与粗尺度 HOG 分支，最终选中局部对比度联合特征；第四轮复用第二轮模型，改用十视图分数平均；第五轮使用人工参考标签作为开发集校准决策阈值；第六轮进一步联合选择 HOG 权重、核参数与阈值。

## 1. 实验结果与评价口径

| 项目 | 第一轮 `svm_run` | 第二轮 `svm_shift_run` | 第三轮 `svm_local_multiscale_run` | 第四轮 `svm_multiview_run` |
|---|---|---|---|---|
| 特征 | 对比度 + 细 HOG | 对比度 + 细 HOG | 对比度 + 细 HOG + 0.5 × 局部对比度 | 与第二轮相同 |
| 训练增强 | 原图 + 镜像 | 五种位置，各自镜像 | 与第二轮相同 | 复用第二轮全量模型 |
| 最终模型训练视图数 | 1,600 | 8,000 | 8,000 | 同第二轮模型 |
| 选定 C / gamma | 1 / 0.5 | 1 / 2.0 | 1 / 2.0 | 固定 1 / 2.0 |
| 推理平均视图数 | 2 | 2 | 2 | 10 |
| 五折平均准确率 | 100% | 100% | 100% | 99.875% |
| 五折标准差 | 0 | 0 | 0 | 0.0025 |
| 现有报告记载的平台 ACC | 0.879 | 0.910 | 0.895 | 0.900 |
| 测试预测：人脸 / 非人脸 | 69 / 351 | 82 / 338 | 76 / 344 | 78 / 342 |

平台 ACC 为已有本地实验报告记载的提交反馈，本仓库未通过平台接口独立核验。前两轮平台分数相差 **3.1 个百分点**。增强方式与 gamma 同时变化，因此不能把增益全部归因于平移增强，确认各因素的作用需要控制其他设置的消融实验。

**前三轮的 100% 是用于选参的同一套内部交叉验证成绩，不是独立测试准确率。** 多组参数共享同一划分并据此选优，会产生模型选择偏差。历史探索中已经查看过测试图像及人工参考标签，平台反馈也参与方案比较，因此整个探索过程不能称为完全独立的盲测。前四轮实验脚本仅读取原始训练集标签，不读取测试集标注；第五、六轮例外，明确读取外部提供的人工参考标签进行阈值或配置选择。

第二轮还保留事后本地评估汇总：相对人工参考标签为 382/420，约 90.9524%。人工标注并非平台官方真值；四舍五入接近平台分数不能证明逐样本一致。该汇总不含逐样本参考标签，原始参考标签未上传，因此仓库不能重新计算该项评估。

第三轮平台 ACC 为 0.895，比第二轮下降 1.5 个百分点。因此第二轮仍是当前平台成绩最高的已测方案。第三轮最终配置未使用粗尺度 HOG，B、C 候选也未分别提交平台，不能据此断言多尺度 HOG 无效。第三轮不计算本地人工标签测试 ACC。

第四轮固定模型，仅把两视图推理改为十视图平均，报告记载的平台 ACC 为 0.900，低于第二轮 1.0 个百分点。相对第二轮只有 4 个预测翻转（ID 9、20、216、329）；没有平台逐样本真值，不能将它们全部判定为新增错误。其事后人工参考评估汇总为 380/420（90.4762%），不等同于平台官方准确率，原始人工标签仍未发布。

### 第五轮：人工参考标签驱动的阈值校准

| 项目 | `svm_threshold_calibrated_run` |
|---|---|
| 基础模型 | 第二轮镜像 / 平移增强 RBF-SVM，C=1、gamma=2 |
| 特征与推理 | 对比度 + 细 HOG，原图 / 镜像两视图分数平均 |
| 最优阈值 | -0.5；分数严格大于阈值判为人脸 |
| 本地开发集成绩 | 392/420 = 93.3333%，FP=11、FN=17 |
| 平台反馈记录 | 0.938（用户反馈，见 platform_feedback.json） |
| 预测分布 | 人脸 112、非人脸 308 |

**第五轮明确使用了这 420 个样本的人工参考标签进行阈值选择。392/420 是调参开发集成绩，不是独立测试准确率。** 平台反馈 0.938 也发生于同一批输入经过人工标签调参之后，不能将其作为对全新样本泛化能力的无偏估计。相较第二轮记录的 0.910，平台显示分数增加 2.8 个百分点，但评价协议已改变，不能作为纯训练集选参方案的公平独立比较。

第五轮 `results.json` 是提交前记录，保留 `platform_acc: null` 及当时尚未提交的说明；后续反馈单独保存在 `platform_feedback.json`，其 0.938 是当前已记录的反馈，未通过平台接口独立核验。

### 第六轮 联合特征核参数与阈值搜索

第六轮目录为 `svm_target095_run`。在 800 张训练图像的十视图增强上拟合单个 SVM，以 420 个样本的人工参考标签选择 HOG 权重、C、gamma 和阈值。完整比较 27 组模型，最佳为 `hog_1.0, C=20, gamma=2, threshold=-0.44404653928318927`。开发集为 **396/420=94.2857%**（FP=6、FN=18），用户反馈的平台 ACC 为 **0.948**。相较第五轮平台显示成绩增加 1.0 个百分点。

本轮人工标签不仅用于阈值，还用于模型配置选择；开发集成绩及相同输入上的平台反馈均不构成全新样本的独立无偏评价。`target095` 是目标名称，实际开发集和平台成绩都未达到 95%。提交文件 `result_svm_acc09429.txt` 的文件名对应本地分数，不应误读为平台成绩。

本轮包含已检查的四页 Word 实验报告及 Markdown 版本，详见 [实验报告](svm_target095_run/实验报告.md)。

## 2. 仓库结构

```text
.
├── README.md
├── requirements.txt
├── predict.py                         # 使用已保存模型推理
├── data_manifest.json                 # 三个数据文件的 SHA-256、变量与形状
├── VERIFICATION.md                    # 此次整理的实际验证情况
├── train(Task_2)/
│   ├── train_data.mat
│   └── train_label.mat
├── test(Task_2)/
│   └── test_data.mat
├── svm_run/
│   ├── face_model.py                  # 特征、镜像、SVM 类及辅助函数
│   ├── run_svm.py                     # 9 组参数的五折搜索与最终训练
│   ├── rbf_svm.joblib                 # 已训练的第一轮模型
│   ├── results.json                   # 参数搜索与折外评价记录
│   ├── training_oof_validation.txt     # 训练样本折外预测，不是测试真值
│   └── result.txt                     # 第一轮 420 条预测
├── svm_shift_run/
    ├── face_model.py
    ├── run_shift_svm.py               # 18 组参数的五折搜索与最终训练
    ├── model.joblib                   # 第二轮 SVC 与推理配置
    ├── results.json
    ├── cv_progress.json               # 每折、每组参数的验证记录
    ├── local_evaluation_after_platform.json  # 历史事后评估汇总
    └── result_shift_svm.txt            # 第二轮 420 条预测
├── svm_local_multiscale_run/
    ├── face_model.py
    ├── model_features.py              # 局部对比度与粗尺度 HOG
    ├── run_experiment.py              # 36 组特征及核参数的五折搜索
    ├── model.joblib
    ├── results.json
    ├── cv_progress.json
    ├── verification.json              # 原实验的推理核验记录
    └── result_local_multiscale.txt     # 第三轮 420 条预测
├── svm_multiview_run/
    ├── face_model.py
    ├── run_multiview.py                # 固定参数五折检查、十视图推理
    ├── results.json                   # 含复用模型 SHA-256 与翻转编号
    ├── local_evaluation_after_platform.json  # 事后人工参考评估汇总
    └── result_multiview.txt            # 第四轮 420 条预测
├── svm_threshold_calibrated_run/
    ├── face_model.py
    ├── run_calibration.py             # 需自行提供私有人工开发标签
    ├── model_calibrated.joblib         # 第二轮模型 + 校准阈值
    ├── results.json                   # 提交前阈值搜索记录
    ├── platform_feedback.json         # 后续平台反馈记录
    └── result_threshold.txt           # 第五轮 420 条预测
└── svm_target095_run/
    ├── face_model.py
    ├── run_search.py                  # 27 组配置和开发集阈值搜索
    ├── model.joblib
    ├── best_scores.npz                # 仅有 score 与 pred，无人工标签
    ├── results.json
    ├── search_progress.json
    ├── platform_feedback.json
    ├── result_svm_acc09429.txt
    ├── build_report.py                # 额外需要 python-docx
    ├── 实验报告.md
    └── 人脸分类实验报告_ACC0948.docx
```

本仓库采用文件白名单整理：测试集只发布 `test_data.mat`；`test_label.mat`、所有 `test_label_before_*.mat`、人工标注目录、缓存、页面截图和其他实验目录均未纳入。保留 `train_label.mat` 是为了支持监督训练。第五轮原始 `scores_and_predictions.npz` 含 `manual_reference` 逐样本标签，已排除且加入忽略规则。原始实验目录未修改。

## 3. 数据格式

训练输入为 `train_data.mat` 中的 `train_data`，形状 `(800, 361)`；训练标签为 `train_label.mat` 中的 `train_label`，共 800 个，400 个人脸和 400 个非人脸。测试输入为 `test_data.mat` 中的 `test_data`，形状 `(420, 361)`。每一行对应一个 19×19 灰度图像，标签 `+1` 表示人脸，`-1` 表示非人脸。

图像向量使用 MATLAB 按列展开顺序。正确恢复单张图像的方法是：

```python
image = row.reshape((19, 19), order='F')
```

实现中使用批量 reshape 后交换行列的方式恢复图像方向。不能直接把默认 C 顺序 reshape 的结果当成正确方向的图像进行水平镜像。`predict.py` 的加载函数支持 `[0, 1]` 或 `[0, 255]` 输入，后者会除以 255；训练脚本按课程原始文件直接加载输入，不自动执行该缩放，换用其他数据时应保持训练与推理预处理一致。

## 4. 环境安装

本次使用 Python 3.12.7 验证，请在仓库根目录建立独立环境。依赖锁定为本次验证使用的版本，不代表已恢复历史训练时的完整环境。

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe predict.py --experiment shift --output prediction.txt
```

Linux / macOS：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python predict.py --experiment shift --output prediction.txt
```

以下命令中的 `python` 指已激活的虚拟环境解释器，也可以直接替换为上述解释器路径。若出现 NumPy / SciPy 二进制版本不兼容错误，请在新建环境中安装本仓库依赖，避免复用存在版本冲突的全局环境。

## 5. 直接使用已保存模型

```bash
python predict.py --experiment baseline --output verification_outputs/baseline.txt
python predict.py --experiment shift --output verification_outputs/shift.txt
python predict.py --experiment multiscale --output verification_outputs/multiscale.txt
python predict.py --experiment multiview --output verification_outputs/multiview.txt
python predict.py --experiment threshold --output verification_outputs/threshold.txt
python predict.py --experiment target095 --output verification_outputs/target095.txt
python predict.py --experiment shift --input "test(Task_2)/test_data.mat" --output prediction.txt
```

无需训练或测试标签。自定义 MAT 文件应只包含一个可识别的二维图像矩阵，支持 `(N, 361)` 或 `(361, N)`；加载器会检查尺寸、数值有限性和像素范围。推理输出目录会自动创建。

`joblib` 文件包含 Python 序列化对象，只应加载可信来源的文件。保存模型中的版本标记为 scikit-learn 1.9.1；本次验证环境为 1.5.2，加载会出现 `InconsistentVersionWarning`。此前四轮模型与第五轮校准模型均成功推理，且 420 条输出分别与各自历史预测逐项一致，但这不保证其他数据上的跨版本兼容性。若遇到问题，可使用仓库代码重新训练；重训模型将使用当前依赖版本。

## 6. 从头运行实验

在仓库根目录执行：

```bash
python svm_run/run_svm.py
python svm_shift_run/run_shift_svm.py
python svm_local_multiscale_run/run_experiment.py
python svm_multiview_run/run_multiview.py
```

前四轮脚本均以脚本位置定位数据，不依赖原电脑的盘符。第一轮已移除历史脚本中指向另一项目的绝对导入路径，以及运行结束时从该路径覆盖 `face_model.py` 的操作；算法与参数搜索逻辑保留。

**运行训练会覆盖对应目录中的模型、预测与 JSON 记录。** 如果需要保留此次发布的历史记录，请先复制仓库或备份这些输出。第二轮会构建增强样本的密集 RBF 核矩阵，对内存需求更高；每折 6,400 个训练视图的一个 float64 核矩阵约占 328 MB，进程实际峰值还包括特征、临时矩阵和 SVM 内存，不应仅据此估算总内存。历史日志记录第二轮运行约 33.54 秒，该时间取决于当时硬件和环境，不能视为当前设备的耗时承诺。

### 6.1 特征处理

1. 每幅图像减去自身像素均值，再进行 L2 归一化，得到 361 维对比度特征。
2. 恢复 19×19 方向后，用 `numpy.gradient` 计算梯度。
3. 使用 `[0, π)` 无符号方向，9 个方向桶，在相邻方向桶间线性分配梯度幅值。
4. 取 4×4 个 cell，每个 cell 为 4×4 像素，对应图像索引区间 1 到 16。
5. 使用 2×2 cell 的 block，步长一个 cell，形成 3×3 个 block；每个 block 为 36 维。
6. block 先 L2 归一化、以 0.2 截断、再次归一化；拼接得到 324 维 HOG，再进行整体归一化。
7. 对比度特征与权重为 1.0 的 HOG 拼接后再次 L2 归一化，最终为 **685 维**。

这是仓库自实现的 HOG，具体 cell 边界及归一化步骤由 `face_model.py` 定义，不应假定与其他库的默认 HOG 完全等价。文件保留其他历史特征模式与辅助类，前两轮实际使用的均为 `hog_1.0`；第三轮以其两个分支为基础扩展特征，第四轮沿用第二轮特征。六轮最终均为单个 SVM，均非模型集成。

### 6.2 第一轮：镜像增强

- 使用 `StratifiedKFold(5, shuffle=True, random_state=20261110)`。
- 搜索 `C ∈ {1, 5, 20}`、`gamma ∈ {0.5, 2, 8}`，共 9 组。
- 先按原始样本划分训练折与验证折，再只在训练折内追加镜像。
- 原图和镜像各分配原样本一半权重，保持每个原样本总权重为 1。
- 验证和测试时对原图、镜像的 decision score 求均值，以 0 为阈值，严格大于 0 判为人脸。
- 按五折平均准确率选择参数，并列时保留搜索顺序中的第一组，最终得到 `C=1, gamma=0.5`。

### 6.3 第二轮：镜像与平移增强

- 使用与第一轮相同的五折划分和随机种子。
- 五种视图：原位置、上、下、左、右各移动 1 像素，边缘使用复制填充；每种视图再做水平镜像，共 10 种。
- 搜索 `C ∈ {1, 5, 20}`、`gamma ∈ {0.5, 2}`、人脸权重 `∈ {1, 1.5, 2}`，共 18 组。
- 特征计算是逐图确定性处理，可以预先缓存；训练核矩阵仅使用当前训练折原样本对应的 10 种视图，验证原样本的增强视图不会进入该折训练。
- 每个视图权重为类别权重的 1/10；非人脸类别权重固定为 1。
- 同一折同一 gamma 的核矩阵可供多组 C 和类别权重复用；交叉验证采用预计算核，最终保存模型采用直接 RBF 核。
- 验证和推理仍只平均原图与镜像的分数，不对平移视图做测试时平均。
- 相同均值按预定义搜索顺序选取，最终为 `C=1, gamma=2, face_weight=1`。

### 6.4 第三轮：局部对比度与多尺度候选比较

保持第二轮的十视图增强、按原样本分组的五折划分、零阈值和原图 / 镜像推理分数平均。新增两个特征分支：

- 局部对比度：用 sigma=1.5 的高斯滤波估计局部均值与方差，计算 `(image - mean) / sqrt(variance + 0.03²)`，截断到 `[-3, 3]` 后展开并 L2 归一化，得到 361 维。
- 粗尺度 HOG：2×2 个 cell，每个 cell 为 8×8 像素，9 个无符号方向桶，形成 36 维向量，L2 归一化、0.2 截断后再次归一化。

候选特征先按分支独立归一化，新增分支乘以 `extra_weight`，拼接后再整体归一化：

| 方案 | 分支 | 维度 |
|---|---|---|
| A | 全局对比度 + 细 HOG + 局部对比度 | 1046 |
| B | 全局对比度 + 细 HOG + 粗 HOG | 721 |
| C | 全局对比度 + 细 HOG + 局部对比度 + 粗 HOG | 1082 |

搜索 A/B/C、`extra_weight ∈ {0.5, 1}`、`C ∈ {1, 5, 20}`、`gamma ∈ {0.5, 2}`，共 36 组，每组五折。每个增强视图权重为 0.1。并列时按上述固定遍历顺序保留首个最优配置。最终选中 **A、extra_weight=0.5、C=1、gamma=2**，1046 维；最终模型没有粗尺度 HOG。`results.json` 保留全部候选成绩，历史耗时约 129.56 秒，不代表其他环境的耗时。

### 6.5 第四轮：固定模型的十视图推理

第四轮复用 `svm_shift_run/model.joblib`，没有新增独立全量模型文件。输入特征、C=1、gamma=2、阈值 0 和模型权重保持不变，只将推理从原图 / 镜像两个视图扩展为原位置、上下左右各平移 1 像素及各自镜像，共十个视图，等权平均未经概率校准的 SVM decision score。属于单模型测试时增强，不是多模型集成。`predict.py --experiment multiview` 可以直接推理而不重跑验证。

`run_multiview.py` 先按原样本分组进行五折检查，每折仅在训练折的十视图上拟合固定参数 SVM，再分别计算两视图与十视图验证结果。十视图折准确率为 `[1, 0.99375, 1, 1, 1]`，平均 0.99875、标准差 0.0025；两视图对照五折均为 1。这里没有重新搜索参数，固定参数来自前轮选择，因此这一验证也不是独立外部测试。

最终测试预测直接复用第二轮全量模型，模型 SHA-256 为 `c0fc15da95ffc2707a6945b7d68a8e67ec9de6453c7bc5c316f8610f0454b624`。本轮 `results.json` 的 `model_source` 保留历史电脑绝对路径作为来源记录；实际代码通过脚本相对路径定位仓库内模型，不依赖该绝对路径。运行本轮脚本需要保留第二轮目录及模型、预测文件，并会覆盖本轮结果与日志；不会覆盖第二轮模型。历史日志耗时约 73.67 秒。

训练与推理脚本不读取测试标签。`local_evaluation_after_platform.json` 是另行生成的事后人工参考汇总，未包含逐样本真值；现有报告生成源码早于这份汇总，其“本轮不计算本地 ACC”描述对应先前执行阶段。本仓库以实际保留的两阶段记录区分训练 / 推理与事后评估，避免混用。

### 6.6 第五轮：开发集阈值选择

保持第二轮 SVM 权重、特征、原图 / 镜像分数平均不变，枚举阈值 `[0, -0.05, -0.1, -0.15, -0.2, -0.25, -0.3, -0.4, -0.5, -0.6, -0.8]`。根据人工开发标签的正确数最大化选择阈值，同分保留遍历中的第一项，选中 -0.5。开发集混淆矩阵为 `[[291, 11], [17, 101]]`，行和列的类别顺序均为 `[-1, 1]`。这一步不是概率校准，也没有重训 SVM。

直接使用校准模型无需标签：

```bash
python predict.py --experiment threshold --output prediction.txt
```

若要重新执行阈值选择，需要自行提供包含 `test_label` 变量的私有 MAT 标签文件，标签顺序应与 420 行输入一致：

```bash
python svm_threshold_calibrated_run/run_calibration.py --labels "PATH/TO/private_labels.mat"
```

仓库不分发该人工标签，所以仅凭公开文件无法重新计算开发集准确率或复现阈值搜索。仓库副本将原脚本硬编码的标签路径改为显式必填 `--labels`，并移除 NPZ 输出中的 `manual_reference` 字段；可生成的 NPZ 仅含 ID、模型分数和预测，仍默认忽略上传。原始 NPZ 完全未上传。运行校准会覆盖本轮模型、结果和提交前 JSON，但不会自动更新已有 `platform_feedback.json`，修改后需要另行核验反馈对应的提交文件。

`results.json` 中模型 / 人工标签的 SHA-256 与历史绝对路径作为来源信息保留；模型可直接加载，绝对路径不参与当前仓库运行。开发标签校验值不等于发布标签本身。

### 6.7 第六轮 联合搜索

- 特征模式按 `hog_1.0, hog_2.0, hog_0.5` 遍历；685 维联合特征中 HOG 的相对权重分别为 1、2、0.5。
- 每个模式依次搜索 `(C,gamma)=(5,2),(20,2),(5,0.5),(20,0.5),(1,0.5),(1,2),(1,8),(5,8),(20,8)`，共 27 组。
- 每个模型根据开发集分数，在相邻唯一分数的中点、两端外侧与 0 中选择阈值；正确数并列时取绝对值最小的阈值，模型配置并列则保留先出现者。
- 拟合仅使用训练集十视图，每视图权重 0.1；推理为原图与镜像两视图分数平均。没有五折验证。
- 原设开发集达到 95% 时提前停止，本次未触发；历史运行时间约 636.47 秒。

```bash
python predict.py --experiment target095 --output prediction.txt
python svm_target095_run/run_search.py --labels "PATH/TO/private_labels.mat"
```

直接推理不需要标签，完整选参需要私有人工开发标签。仓库版将硬编码标签路径改为显式 `--labels`，并修正原脚本只有达到目标才输出预测的条件，搜索结束后始终导出 `result_svm_acc09429.txt`。文件名保留本轮历史名称，未来重跑所得分数应以新日志为准。运行会覆盖模型、日志及预测；旧平台反馈文件不会自动同步。

`best_scores.npz` 已检查仅包含 `score`、`pred`，可以公开；与第五轮被排除的含人工标签 NPZ 不同。重建 Word 报告需额外安装 `python-docx`，再执行 `python svm_target095_run/build_report.py`。报告中的平台反馈来源为本次用户明确提供的 0.948。

## 7. 预测文件与记录说明

预测文件为无表头的两列整数：

```text
1 -1
2 1
3 -1
```

这里只是格式示例，不代表前三个样本的实际预测。第一列是从 1 开始的样本 ID，第二列是 `-1` 或 `1`。对仓库测试集应输出 420 行，行顺序与 MAT 矩阵行顺序一致。`predict.py` 和第二轮脚本显式写入 ASCII 与 LF 换行。第一轮原始 `result.txt` 保留历史文件字节，可能使用 Windows 换行。

Git 通过 `.gitattributes` 将仓库文本统一规范为 LF，因此 GitHub 文件的换行字节可能与原始本地文件不同，预测整数内容保持一致。

仓库不补造第 421 到 1000 个样本，也不将平台最终接受格式作为已确认事实；实际提交应遵循平台当前要求。`results.json` 保留候选配置及折准确率，第一轮还包含所选模型的折外混淆矩阵，类别顺序为 `[-1, 1]`。预测类别数量是模型输出统计，不是测试集真实类别比例。

## 8. 数据来源与使用范围

数据来自本课程下发材料，本次按原数据文件发布。未发现明确的数据集再分发许可证，因此仓库不额外声称拥有数据版权，也未擅自为课程数据附加开源许可证。使用、转载或再分发时需确认课程材料的授权条件。

本项目用于课程实验复现和方法学习。第五、六轮使用人工标签调参，应与前四轮的评价协议分别理解。当前记录没有提供多随机种子、嵌套交叉验证、跨来源测试或统计显著性证据，不能据此推断对其他人脸数据或现实应用具有同样表现。
