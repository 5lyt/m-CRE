# GitHub和Zenodo更新具体流程

这次的 Supplementary_Software_1_mCRE_full.zip 是完整仓库源码包，不是上次的extension。正文投稿包和两份审稿回复已经使用同一套术语、数据编号与结果口径。当前已读取公开仓库，基线commit为 e5a6b45f3967b00da926af471a32b56c87e85a2f，已有公开release v1.0.0；本次未执行push或发布。建议下一版本v1.1.0，执行时先检查是否仍未使用。

## 下载与本地文件准备

1. 解压完整代码ZIP，得到 m-CRE/，里面应直接包含 README.md、LICENSE、code/、R/、analysis/、data/、docs/、results/、provenance/、tools/。不要再把整个ZIP本身上传到code/当作唯一源码。
2. 保留上一轮模型两包和数组两包，原文件未变，不必再次生成。四物种common表与HDF5已在新版源码内。
3. 在代码根目录运行 `python -m pip install -r requirements-audit.txt`，然后 `python analysis/verify_new_materials.py --output-dir verified_new`。完整运行顺序见docs/REPRODUCIBILITY.md。

## 推荐的git上传方式

先在新目录取得仓库，并建修订分支：

```bash
git clone https://github.com/5lyt/m-CRE.git m-CRE-upload
cd m-CRE-upload
git switch -c revision-20261016
```

将解压得到的m-CRE目录与这个checkout放在不同位置。下面的 `/path/to/unpacked/m-CRE` 替换为实际解压目录；Windows可使用带引号的 `D:/mcre_release/m-CRE`：

```bash
python /path/to/unpacked/m-CRE/tools/apply_release.py --source /path/to/unpacked/m-CRE --target . --dry-run
python /path/to/unpacked/m-CRE/tools/apply_release.py --source /path/to/unpacked/m-CRE --target .
git status --short
git diff --stat
git diff -- README.md docs
```

复制工具只处理文件，不删除旧文件，不覆盖.git，不创建commit/push；会核对目标remote、工作区是否干净和LICENSE是否保持一致。审阅后提交本次目录：

```bash
git add README.md LICENSE .gitignore requirements.txt requirements-audit.txt requirements-training.txt code R analysis data docs results/revised provenance tools
git commit -m "Unify Scientific Reports revision code data and reproducibility documentation"
git push -u origin revision-20261016
```

在GitHub打开该分支的Pull request，确认diff后合并main。原始许可为Academic Research License，代码包原样保留，不要改成一个未经作者确认的许可证。

## 创建新的GitHub Release

合并后更新本地main并检查tag：

```bash
git switch main
git pull --ff-only
git fetch --tags
git tag --list v1.1.0
```

若最后一条已有输出，选择下一个未使用版本，不能覆盖原tag。若为空，可以用网页仓库右侧Releases → Draft a new release，Target选main最终commit，创建v1.1.0。标题用“m-CRE v1.1.0”，正文复制docs/RELEASE_NOTES.md。先保存Draft。

装有GitHub CLI也可创建草稿，实际版本号按上一检查确定：

```bash
gh release create v1.1.0 --draft --target main --title "m-CRE v1.1.0" --notes-file docs/RELEASE_NOTES.md
```

给release附上完整代码ZIP、投稿资料包、四份既有模型/数组ZIP及SHA256校验清单。源码文件已经提交到仓库；这些ZIP是下载资产。逐个核对文件后再发布Release。不要把编译日志、临时验证目录、作者内部批注或本机路径写入公开README。

## 仅使用GitHub网页时

在新分支中用Add file → Upload files，按code、R、analysis、data、docs等目录分批拖入文件，并确认目录结构。网页单文件限制25 MiB、每批最多100文件，完整源码有数百文件，所以建议上面的git方式。模型和数组ZIP放在Release附件。网页上传后同样通过PR合并并创建新Release。

## 在现有Zenodo记录创建新版本

原记录：https://zenodo.org/records/20505253 ，原专属DOI：https://doi.org/10.5281/zenodo.20505253 。

1. 用有管理权限的账户打开现有记录，选择New version。先确认原记录是否由GitHub集成产生；若是，优先沿用该集成，不要同时手工另建重复系列。
2. 新版本草稿中保留仍使用的旧文件，替换旧代码包，上传完整代码、必要数据资产和校验清单。不能只改旧记录的文字说明就声称归档了新文件。
3. Version与GitHub新tag一致；作者及许可沿用原记录并由作者确认。相关标识填写本次GitHub release页面。可在草稿预留新DOI，但预留不等于已经发表。
4. 若预留了新DOI，先把论文、README、回复和引用信息同步，再上传最终文件；核对下载后发布Zenodo新版本。
5. 记录新版本的专属DOI及record URL。20505253仍表示原v1.0.0，不要把它配成v1.1.0的专属DOI。

## 发布后必须同步的三处

当前投稿稿已用“revised source provided as Supplementary Software 1”准确描述交付，未虚构新公开版本。实际发布后，用真实值替换如下模板中的三个标记，并同步：

- 正文Data availability和参考文献mcregithub条目；
- 完整审稿回复中Reviewer 4的Major comment 2、Comment 2.3和DA1引用段；
- 根README、provenance/repository_origin.json及发布说明。

英文模板：

The source code, full common feature tables, candidate metadata, stored motif patterns and analysis scripts supporting this revision are available in the m-CRE GitHub repository (https://github.com/5lyt/m-CRE; release ACTUAL_NEW_TAG) and archived in Zenodo (DOI ACTUAL_NEW_VERSION_DOI; record ACTUAL_NEW_RECORD_URL). Reproduction commands, data-asset checksums and the scope of the supplied results are described in the repository documentation.

只有实际发布/可下载的资产才能写为公开。若新DOI还未发布，继续使用稿件当前的Supplementary Software交付表达即可，不写不存在的DOI。

官方步骤依据：
- https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository
- https://cli.github.com/manual/gh_release_create
- https://help.zenodo.org/docs/deposit/manage-versions/
