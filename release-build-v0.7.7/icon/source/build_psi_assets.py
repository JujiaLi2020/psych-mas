from pathlib import Path
import subprocess, json, shutil, hashlib, zipfile
from PIL import Image, ImageDraw, ImageFont
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.backends.backend_svg import _short_float_fmt
ROOT=Path('/workspace/scratch/1301238a86f6/output/PsyMAS_Psi_Brand_Kit_v1')
for d in ['svg','png','desktop','web','docs','source']: (ROOT/d).mkdir(parents=True,exist_ok=True)
NAVY='#17354D'
# Unified closed outline; bilateral symmetry, equal terminals, short stem.
D='M72 64 H132 Q140 64 140 72 V224 Q140 276 192 276 H218 V72 Q218 64 226 64 H286 Q294 64 294 72 V276 H320 Q372 276 372 224 V72 Q372 64 380 64 H440 Q448 64 448 72 V224 Q448 352 320 352 H294 V432 Q294 440 286 440 H226 Q218 440 218 432 V352 H192 Q64 352 64 224 V72 Q64 64 72 64 Z'
def svg(body,w=512,h=512): return f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}"><title>PsyMAS — Test Security Analytics</title>{body}</svg>'
def mark(color=NAVY): return f'<path fill="{color}" d="{D}"/>'
def textpath(text,x,y,size,color,weight='regular'):
 p=TextPath((0,0),text,size=size,prop=FontProperties(family='DejaVu Sans',weight=weight))
 seg=[]
 for verts,code in p.iter_segments(curves=True):
  vals=' '.join(f'{v:.4f}' for v in verts)
  seg.append({1:'M',2:'L',3:'Q',4:'C',79:'Z'}[code]+('' if code==79 else vals))
 return f'<path fill="{color}" transform="translate({x} {y}) scale(1 -1)" d="{" ".join(seg)}"/>',p.get_extents().width
files={}
for name,col in [('navy',NAVY),('white','#FFFFFF'),('black','#000000')]:
 files[f'mark-{name}']=svg(mark(col))
 for layout in ['horizontal','stacked']:
  if layout=='horizontal':
   b=f'<g transform="translate(0 0) scale(.5)">{mark(col)}</g>'
   b+=textpath('PsyMAS',280,136,94,col,'bold')[0]
   b+=textpath('Test Security Analytics',284,187,28,col)[0]
   files[f'logo-{layout}-{name}']=svg(b,780,256)
  else:
   b=f'<g transform="translate(128 0)">{mark(col)}</g>'
   for t,y,size,weight in [('PsyMAS',558,98,'bold'),('Test Security Analytics',610,30,'regular')]:
    _,tw=textpath(t,0,0,size,col,weight)
    b+=textpath(t,(768-tw)/2,y,size,col,weight)[0]
   files[f'logo-{layout}-{name}']=svg(b,768,672)
files['app-icon']=svg(f'<rect width="512" height="512" rx="104" fill="{NAVY}"/><g transform="translate(41 41) scale(.84)">{mark("#FFFFFF")}</g>')
files['app-icon-maskable']=svg(f'<rect width="512" height="512" fill="{NAVY}"/><g transform="translate(77 77) scale(.70)">{mark("#FFFFFF")}</g>')
for name,s in files.items(): (ROOT/'svg'/f'{name}.svg').write_text(s)
def render(name,path,width):
 subprocess.run(['inkscape',str(ROOT/'svg'/f'{name}.svg'),'--export-type=png',f'--export-filename={path}',f'--export-width={width}'],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
for name in files:
 render(name,ROOT/'png'/f'{name}-1024.png',1024)
for size in [16,20,24,32,40,48,64,128,256,512]:
 for name in ['mark-navy','mark-white','app-icon']:
  im=Image.open(ROOT/'png'/f'{name}-1024.png'); im.resize((size,size),Image.Resampling.LANCZOS).save(ROOT/'desktop'/f'{name}-{size}.png')
for name in ['app-icon','mark-navy','mark-white']:
 Image.open(ROOT/'png'/f'{name}-1024.png').resize((256,256),Image.Resampling.LANCZOS).save(ROOT/'desktop'/f'{name}.ico',sizes=[(s,s) for s in [16,24,32,48,64,128,256]])
shutil.copy(ROOT/'desktop/app-icon.ico',ROOT/'web/favicon.ico')
shutil.copy(ROOT/'svg/app-icon.svg',ROOT/'web/favicon.svg')
for size,fn,name in [(180,'apple-touch-icon.png','app-icon-maskable'),(192,'icon-192.png','app-icon'),(512,'icon-512.png','app-icon'),(512,'icon-maskable-512.png','app-icon-maskable')]:
 Image.open(ROOT/'png'/f'{name}-1024.png').resize((size,size),Image.Resampling.LANCZOS).save(ROOT/'web'/fn)
(ROOT/'web/site.webmanifest').write_text(json.dumps({'name':'PsyMAS','short_name':'PsyMAS','start_url':'./','display':'standalone','background_color':'#FFFFFF','theme_color':NAVY,'icons':[{'src':'icon-192.png','sizes':'192x192','type':'image/png','purpose':'any'},{'src':'icon-512.png','sizes':'512x512','type':'image/png','purpose':'any'},{'src':'icon-maskable-512.png','sizes':'512x512','type':'image/png','purpose':'maskable'}]},indent=2))
(ROOT/'web/head-snippet.html').write_text('''<!-- Copy web/ contents to your static root; adapt URLs if served below a subpath. -->
<link rel="icon" href="/favicon.ico" sizes="any">
<link rel="icon" href="/favicon.svg" type="image/svg+xml">
<link rel="apple-touch-icon" href="/apple-touch-icon.png">
<link rel="manifest" href="/site.webmanifest">
<meta name="theme-color" content="#17354D">
''')
(ROOT/'README_使用说明.md').write_text('''# PsyMAS Ψ 资源包 v1

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

PyInstaller 的应用图标参数示例：`--icon assets/brand/desktop/app-icon.ico`。运行时窗口图标与安装器图标仍需分别在现有框架、安装器配置中指定，不能只改打包图标。Inno Setup 可在 [Setup] 使用 `SetupIconFile=assets\\brand\\desktop\\app-icon.ico`，路径相对脚本目录。

### 网站

复制 web/ 文件到静态资源目录，参考 web/head-snippet.html。若部署在子路径，调整 href 与 manifest 路径。替换 favicon 后可能需要清除浏览器缓存。manifest 本身不提供离线功能。

这些是接入示例，尚未在 PsyMAS 当前代码库中运行或修改程序。

## 使用规则

主标志优先使用深蓝；深色背景使用纯白。避免渐变、阴影、拉伸、旋转、增加节点或中心色块。图形已有留白，不裁到边缘。副标题小于可读尺寸时移除，只保留图标或名称。托盘必须按实际系统主题选择有对比度的版本。

## 来源与维护

这是可编辑几何重建版本，不宣称与生成草图逐像素相同。字体使用 DejaVu Sans，最终图形已转路径，授权文件见 docs/FONT_LICENSE.txt。source/build_psi_assets.py 为构建源；需要 Python、Pillow、Matplotlib 与 Inkscape，运行前修改输出目录。文件校验值见 SHA256SUMS.txt。
''',encoding='utf-8')
fontpath=Path(findfont(FontProperties(family='DejaVu Sans')))
licenses=[Path('/usr/share/doc/fonts-dejavu-core/copyright'),fontpath.parent/'LICENSE_DEJAVU']
for f in licenses:
 if f.exists(): shutil.copy(f,ROOT/'docs/FONT_LICENSE.txt'); break
else: raise RuntimeError('Missing font license')
html='''<!doctype html><meta charset="utf-8"><title>PsyMAS Brand Preview</title><style>body{font:16px system-ui;margin:40px;color:#17354D;background:#edf1f4}.card{background:white;padding:32px;margin:20px 0;border-radius:16px}.dark{background:#17354D;color:white}img{vertical-align:middle;margin:12px}h1{font-size:24px}</style><h1>PsyMAS Ψ — Asset Preview</h1>'''
html+='<div class="card"><img src="svg/logo-horizontal-navy.svg" width="580"></div><div class="card dark"><img src="svg/logo-horizontal-white.svg" width="580"></div>'
html+='<div class="card">'+''.join(f'<img src="desktop/mark-navy-{s}.png" width="{s}" height="{s}" title="{s}px">' for s in [16,20,24,32,48,64,128])+'</div>'
html+='<div class="card dark">'+''.join(f'<img src="desktop/mark-white-{s}.png" width="{s}" height="{s}" title="{s}px">' for s in [16,20,24,32,48,64,128])+'</div>'
(ROOT/'preview.html').write_text(html)
# Contact sheet for actual raster QA.
sheet=Image.new('RGB',(1400,960),'#EDF1F4'); draw=ImageDraw.Draw(sheet)
for x,col,name in [(0,'#FFFFFF','navy'),(700,NAVY,'white')]:
 draw.rectangle((x,0,x+699,960),fill=col)
 im=Image.open(ROOT/'png'/f'logo-stacked-{name}-1024.png'); im.thumbnail((600,540)); sheet.paste(im,(x+(700-im.width)//2,20),im)
 for i,s in enumerate([16,24,32,48,64,128]):
  im=Image.open(ROOT/'desktop'/f'mark-{name}-{s}.png'); sheet.paste(im,(x+35+i*108,620),im)
 im=Image.open(ROOT/'desktop/app-icon-128.png'); sheet.paste(im,(x+280,790),im)
sheet.save(ROOT/'docs/asset-preview.png')
shutil.copy(__file__,ROOT/'source/build_psi_assets.py')
# Verify PNG dimensions, SVG parsed XML, and ICO embedded sizes.
import xml.etree.ElementTree as ET
for f in (ROOT/'svg').glob('*.svg'): ET.parse(f)
for f in ROOT.rglob('*.png'):
 with Image.open(f) as im: im.verify()
for f in (ROOT/'desktop').glob('*.ico'):
 im=Image.open(f); assert (16,16) in im.ico.sizes() and (256,256) in im.ico.sizes()
(ROOT/'SHA256SUMS.txt').write_text('\n'.join(hashlib.sha256(f.read_bytes()).hexdigest()+'  '+str(f.relative_to(ROOT)) for f in sorted(ROOT.rglob('*')) if f.is_file() and f.name!='SHA256SUMS.txt')+'\n')
zipname=ROOT.parent/(ROOT.name+'.zip')
with zipfile.ZipFile(zipname,'w',zipfile.ZIP_DEFLATED) as z:
 for f in ROOT.rglob('*'):
  if f.is_file(): z.write(f,f.relative_to(ROOT.parent))
print(json.dumps({'zip':str(zipname),'files':len([p for p in ROOT.rglob('*') if p.is_file()]),'bytes':zipname.stat().st_size}))
