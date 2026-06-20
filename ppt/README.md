# SUSTech Beamer Presentation Template

SUSTech 风格的 Beamer 演示文稿模板，基于 XeLaTeX 编译。

## 编译

```bash
cd template
xelatex main.tex
xelatex main.tex   # 第二次修复交叉引用
```

## 结构

```
template/
├── main.tex              # 主文件（修改此文件即可）
├── figures/
│   ├── Sustech.png       # SUSTech logo（标题页 + 每页右上角）
│   └── background.png    # 标题页背景图
└── README.md
```

## 自定义命令

| 命令 | 用途 | 示例 |
|------|------|------|
| `\tagbox{color}{text}` | 彩色标签 | `\tagbox{Blue}{New}` |
| `\metric{color}{value}{label}` | 指标卡片 | `\metric{Green}{92.3}{Accuracy}` |
| `\smallnote{text}` | 灰色注释 | `\smallnote{Source: xxx}` |

## 颜色

模板预定义了以下颜色：`DeepNavy`, `Ink`, `Muted`, `Blue`, `Teal`, `Orange`, `Red`, `Green`, `Panel`, `Line`, `SoftBlue`。

表格高亮推荐：`\rowcolor{Green!15}`（最优）、`\rowcolor{Red!10}`（警告）。

## 注意事项

- 需要安装 **Arial** 和 **Microsoft YaHei** 字体
- 需要 XeLaTeX 编译器（TeX Live 或 MiKTeX）
- 背景图可通过替换 `figures/background.png` 自定义
