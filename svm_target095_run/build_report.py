from pathlib import Path
import json
from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

r = Path(__file__).resolve().parent
stats = json.loads((r / 'results.json').read_text(encoding='utf-8'))
doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21), Cm(29.7)
sec.top_margin = sec.bottom_margin = Cm(2)
sec.left_margin = sec.right_margin = Cm(2.3)
for name in ['Normal', 'Title', 'Subtitle', 'Heading 1', 'Heading 2']:
    st = doc.styles[name]
    st.font.name = 'Times New Roman'
    st.font.color.rgb = RGBColor(0, 0, 0)
    st.element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'), '宋体' if name == 'Normal' else '黑体')
doc.styles['Normal'].font.size = Pt(10.5)
doc.styles['Normal'].paragraph_format.line_spacing = 1.2
doc.styles['Normal'].paragraph_format.space_after = Pt(6)
for name, size in [('Title', 20), ('Heading 1', 14), ('Heading 2', 11)]:
    doc.styles[name].font.size = Pt(size)
    doc.styles[name].paragraph_format.space_before = Pt(10)
    doc.styles[name].paragraph_format.space_after = Pt(6)
md = []

def p(text):
    doc.add_paragraph(text)
    md.append(text + '\n')

def h(text):
    doc.add_heading(text, 1)
    md.append('## ' + text + '\n')

def table(headers, rows):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = 'Table Grid'
    for c, txt in zip(t.rows[0].cells, headers):
        c.text = txt
        shade = OxmlElement('w:shd'); shade.set(qn('w:fill'), 'EDEDED'); c._tc.get_or_add_tcPr().append(shade)
        for run in c.paragraphs[0].runs: run.bold = True
    repeat = OxmlElement('w:tblHeader'); t.rows[0]._tr.get_or_add_trPr().append(repeat)
    for row in rows:
        for c, txt in zip(t.add_row().cells, row): c.text = str(txt)
    for row in t.rows:
        cant = OxmlElement('w:cantSplit'); row._tr.get_or_add_trPr().append(cant)
        for c in row.cells:
            for par in c.paragraphs:
                par.paragraph_format.space_after = Pt(3)
                par.paragraph_format.space_before = Pt(3)
                for run in par.runs: run.font.size = Pt(10)
    md.extend(['| ' + ' | '.join(headers) + ' |', '|' + '|'.join(['---'] * len(headers)) + '|'])
    md.extend('| ' + ' | '.join(map(str, row)) + ' |' for row in rows)
    md.append('')

def page(): doc.add_page_break()

title = '人脸分类联合参数搜索与阈值校准实验报告'
doc.add_paragraph(title, 'Title'); md.append('# ' + title + '\n')
p('模式识别第四次大作业    实验日期 2026年11月14日')
p('本实验在对比度与 HOG 联合特征的单个 RBF-SVM 上，同时比较特征权重、核参数和分类阈值。SVM 的拟合仅使用课程提供的 800 个训练样本及其增强版本，420 个样本的人工参考标签用于选择配置和阈值，因而这部分数据在本轮承担开发集作用。')
p('最终选中 HOG 权重 1、C=20、γ=2、阈值 −0.4440465393。本地开发集正确 396/420，ACC=94.2857%；平台反馈 ACC=0.948，即 94.8%。相较上一轮平台显示成绩 0.938 提升 1.0 个百分点，但未达到原定 95% 目标。开发集成绩与平台成绩分别报告，不将调参后的本地成绩视为独立测试结果。')
h('1 实验目的与数据')
p('上一轮固定 C=1、γ=2，仅在有限阈值列表中校准。本轮扩大搜索范围，检验改变 HOG 相对权重与 RBF-SVM 核参数，并对每组模型搜索决策阈值，能否进一步改善当前开发集及平台表现。')
table(['数据部分', '规模与用途'], [
    ['原始训练输入', '800×361，每行对应 19×19 灰度图像'],
    ['训练标签', '400 个人脸与 400 个非人脸，用于 SVM 拟合'],
    ['开发输入与人工参考标签', '420 个样本，人工参考为 118 个人脸与 302 个非人脸，用于选参和阈值搜索'],
    ['标签约定与像素', '+1 为人脸，−1 为非人脸；像素范围 [0,1]'],
    ['平台评价', '用户反馈提交文件的 ACC=0.948，不提供平台逐样本真值']])
p('图像按 MATLAB 列优先顺序展开。恢复图像方向后再进行镜像与平移，避免把转置后的图像误当成原方向。人工参考标签不等同于官方平台真值。')

page(); h('2 特征与训练方法')
p('全局对比度分支对每幅图像减去自身均值，再进行 L2 归一化，得到 361 维向量。HOG 分支使用 9 个无符号方向桶，4×4 个 cell，每个 cell 为 4×4 像素；2×2 cell 组成 block，步长为一个 cell，共 9 个 block。各 block 归一化、以 0.2 截断后再次归一化，拼接得到 324 维 HOG。')
p('将对比度向量 z 与乘以权重 w 的 HOG 向量 h 拼接，再整体归一化。特征可记为 φw(x)=normalize([z(x), w·h(x)])，总维度固定为 685；改变 w 调整两个分支的相对贡献。')
p('每张原始训练图像生成原位置、上下左右各平移 1 像素五种版本，每种再生成水平镜像，共十个视图。平移采用边缘像素复制填充。800 个原始样本扩展为 8000 个训练视图，每个视图的权重为 0.1，使每个原样本总权重仍为 1。')
p('分类器采用单个 RBF-SVM，核函数 K(u,v)=exp(−γ‖u−v‖²)。求解设置为 tol=10⁻⁴、max_iter=100000、cache_size=768 MB；每次拟合检查收敛状态。最终模型不是多个 SVM 的集成。预测时仅对原图和水平镜像的决策分数取平均，不对平移视图做平均。')
h('3 联合搜索与阈值规则')
table(['搜索项目', '候选与规则'], [
    ['HOG 权重 w', '依次为 1、2、0.5'],
    ['C 与 γ', 'C∈{1,5,20}，γ∈{0.5,2,8}，每个权重下共 9 对'],
    ['固定遍历顺序', '(5,2)、(20,2)、(5,0.5)、(20,0.5)、(1,0.5)、(1,2)、(1,8)、(5,8)、(20,8)'],
    ['决策阈值', '相邻唯一分数的中点、两端外侧值及 0'],
    ['阈值同分处理', '正确数最大时，优先选择绝对值最小的阈值'],
    ['模型同分处理', '仅正确数严格提高时替换，保留最先达到该分数的配置']])
p('对每个候选模型，用两视图平均分数与人工开发标签计算正确数；严格大于阈值判为人脸，否则判为非人脸。搜索原设开发集 ACC 达到 0.95 时提前停止，本次没有触发停止条件，完成全部 27 组。记录耗时约 636.47 秒，依赖当时硬件与环境。本轮未进行五折或嵌套交叉验证。')

page(); h('4 搜索结果与最终模型')
rows=[]
for mode in ['hog_1.0','hog_2.0','hog_0.5']:
    b=max((t for t in stats['trials'] if t['mode']==mode),key=lambda t:t['correct'])
    rows.append([mode, f"{b['C']:g}", f"{b['gamma']:g}",f"{b['threshold']:.6f}",f"{b['correct']}/420",f"{b['acc']*100:.4f}%"])
table(['特征模式','C','γ','阈值','正确数','开发集 ACC'],rows)
p('最佳配置为 hog_1.0、C=20、γ=2、阈值 −0.44404653928318927。相较上一轮仅校准阈值，本轮既改变了 SVM 的 C，也改用基于全部唯一分数边界的阈值候选，不能把增益全部归因于某一项变化。')
table(['最终结果', '数值与含义'], [
    ['开发集 ACC', '396/420=94.2857%，参数与阈值选择后的成绩'],
    ['平台 ACC', '0.948=94.8%，用户反馈的提交成绩'],
    ['预测分布', '人脸 106，非人脸 314，不是真实类别比例'],
    ['人工参考下的误差', 'FP=6，FN=18，共 24 个错误'],
    ['目标达成情况', '本地与平台均未达到 95%']])
p('下表仅依据人工参考标签计算。行表示参考类别，列表示模型预测类别，不是平台官方混淆矩阵。')
table(['参考类别', '预测非人脸 −1', '预测人脸 +1'], [['非人脸 −1', '296', '6'], ['人脸 +1', '18', '100']])
p('人工参考口径下，人脸 Precision=100/106=94.3396%，Recall=100/118=84.7458%，F1=89.2857%。这些指标同样来自参与模型选择的开发集，不是独立性能估计。平台 ACC 与本地 ACC 相差约 0.5143 个百分点，不能据此反推出官方逐样本答案。')
h('5 与上一轮的比较')
table(['比较项','上一轮阈值校准','本轮联合搜索'], [['C 与 γ','1 与 2','20 与 2'], ['HOG 权重','1','1'], ['阈值','−0.5','约 −0.44405'], ['本地开发集正确数','392/420','396/420'], ['平台反馈 ACC','0.938','0.948']])

page(); h('6 结果解释与局限')
p('本轮开发集正确数增加 4，平台显示成绩增加 1.0 个百分点，说明在当前实验记录中，扩大联合搜索获得了更高的已测分数。目录名 target095 表示搜索目标，不表示达到 95%；提交文件名中的 acc09429 对应本地约 94.29%，平台反馈应单独记为 94.8%。')
p('尽管 SVM 拟合阶段没有把 420 个人工标注样本并入训练，但其特征权重、C、γ 和阈值都根据这些标签选取。开发集已参与模型开发，不能称为未接触的测试集。反复比较配置会使当前开发集成绩偏乐观；平台使用相同输入，也不能据此直接推断对全新样本的泛化能力。')
p('本轮没有多随机种子、独立保留集或统计显著性检验。人工标签与平台真值可能存在差异；本地混淆矩阵只能分析人工参考口径下的错误。目前人工参考下仍有 18 个人脸漏检，是否对其他来源人脸同样如此，需要独立数据验证。')
p('后续可将人工标注样本划分为开发部分与一次性评估部分，或在训练数据内部使用嵌套交叉验证选择特征与核参数，再用单独校准集确定阈值。应在方案冻结后进行独立评价，避免继续根据同一批标签追逐目标分数。')
h('7 文件与复现说明')
table(['文件', '用途'], [['run_search.py','拟合候选 SVM 并以私有开发标签选择配置与阈值'], ['face_model.py','对比度 HOG 特征与阈值选择函数'], ['model.joblib','保存的单个最佳 SVM、特征模式与阈值'], ['results.json 与 search_progress.json','全部 27 组历史搜索记录'], ['best_scores.npz','仅含最佳模型分数与预测，不含人工参考标签'], ['result_svm_acc09429.txt','420 行 ID 与标签，平台提交文件'], ['platform_feedback.json','用户反馈的平台 ACC 0.948']])
p('本次重新加载保存模型，420 条预测与提交文件逐项一致；最佳阈值、396/420 及 FP/FN 已用本机人工参考标签复算。没有重新训练全部 27 组候选，历史候选成绩仍以日志为依据。保存模型跨 scikit-learn 版本加载时存在版本警告，本次仅验证指定输入的一致性。')
p('公开仓库不发布人工测试标签。仓库版搜索脚本通过必填 --labels 接收私有 MAT 文件，并修正原脚本只有达到 95% 才导出预测的问题，使搜索结束后始终导出当前最佳结果；算法搜索与阈值规则保持不变。仅凭公开文件可以直接推理，但不能重新计算开发集分数或完整复现有标签选参。')
h('8 实验结论')
p('通过 HOG 权重、核参数与阈值联合选择，本轮获得 C=20、γ=2、HOG 权重 1 的单个 RBF-SVM。开发集 ACC 为 94.2857%，平台反馈 ACC 为 94.8%，较上一轮提高 1.0 个百分点，仍未达到 95%。结果证明本次配置在当前记录中的分数更高，其独立泛化性能仍需另行验证。')

out = r / '人脸分类实验报告_ACC0948.docx'
for borders in list(doc.styles.element.iter(qn('w:pBdr'))):
    borders.getparent().remove(borders)
for borders in list(doc.element.iter(qn('w:pBdr'))):
    borders.getparent().remove(borders)
doc.save(out)
(r / '实验报告.md').write_text('\n'.join(md), encoding='utf-8')
print(out)
