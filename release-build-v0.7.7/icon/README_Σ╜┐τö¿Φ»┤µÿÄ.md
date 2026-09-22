# PsyMAS Ψ 资源包 v1

本包将确认的 Ψ 草稿重建为统一的矢量几何路径，并非从生成图片中裁切。保留平齐顶端、短下轴、对称轮廓和深蓝单色。标准色 #17354D；白色 #FFFFFF。辅助青绿 #159C91 可用于界面，不添加到本标志中心。

## 文件选择

| 场景 | 文件 |
|---|---|
| Windows 安装程序、桌面快捷方式、应用窗口 | desktop/app-icon.ico |
| 浅色托盘背景 | desktop/mark-navy.ico 或对应尺寸 PNG |
| 深色托盘背景 | desktop/mark-white.ico 或对应尺寸 PNG |
| 网站标签页 | web/favicon.ico 与 web/favicon.svg |
| 网站、README、报告页眉 | svg/logo-horizontal-navy.svg |
| 启动页、演示封面 | svg/logo-stacked-navy.svg |
| 深色背景品牌展示 | 对应 white.svg 或 white PNG |
| 论文黑白印刷 | 对应 black.svg |
| 仅图标 | svg/mark-navy.svg |
| 手机主屏幕与 PWA | web/ 中 PNG 与 site.webmanifest |

SVG 中所有文字均已转为路径，无需安装字体。PNG 是透明背景；app-icon 有深蓝圆角背景，maskable 为全幅实色背景。desktop 提供 16/20/24/32/40/48/64/128/256/512 px；ICO 内含 16/24/32/48/64/128/256 px。png/ 提供宽度 1024 px 的大图。PNG 不要拉伸改变宽高比。

## 接入

将需要的文件复制到项目 assets/brand/。不要整包放入发布目录：source/、docs/、预览及授权说明按项目需要保留。

### Streamlit（示例）

```python
from pathlib import Path
import streamlit as st
from PIL import Image
BRAND = Path(__file__).resolve().parent / "assets" / "brand"
st.set_page_config(page_title="PsyMAS", page_icon=Image.open(BRAND / "desktop/app-icon-32.png"))
st.sidebar.image(str(BRAND / "png/logo-horizontal-navy-1024.png"))
```

### Python 系统托盘（pystray 示例）

```python
from PIL import Image
import pystray
icon = pystray.Icon("psymas", Image.open("assets/brand/desktop/app-icon-64.png"), "PsyMAS")
# 将现有菜单传给 icon.menu，并沿用原有启动/停止逻辑。
```

### Windows 打包

PyInstaller 的应用图标参数示例：`--icon assets/brand/desktop/app-icon.ico`。运行时窗口图标与安装器图标仍需分别在现有框架、安装器配置中指定，不能只改打包图标。Inno Setup 可在 [Setup] 使用 `SetupIconFile=assets\brand\desktop\app-icon.ico`，路径相对脚本目录。

### 网站

复制 web/ 文件到静态资源目录，参考 web/head-snippet.html。若部署在子路径，调整 href 与 manifest 路径。替换 favicon 后可能需要清除浏览器缓存。manifest 本身不提供离线功能。

这些是接入示例，尚未在 PsyMAS 当前代码库中运行或修改程序。

## 使用规则

主标志优先使用深蓝；深色背景使用纯白。避免渐变、阴影、拉伸、旋转、增加节点或中心色块。图形已有留白，不裁到边缘。副标题小于可读尺寸时移除，只保留图标或名称。托盘必须按实际系统主题选择有对比度的版本。

## 来源与维护

这是可编辑几何重建版本，不宣称与生成草图逐像素相同。字体使用 DejaVu Sans，最终图形已转路径，授权文件见 docs/FONT_LICENSE.txt。source/build_psi_assets.py 为构建源；需要 Python、Pillow、Matplotlib 与 Inkscape，运行前修改输出目录。文件校验值见 SHA256SUMS.txt。
