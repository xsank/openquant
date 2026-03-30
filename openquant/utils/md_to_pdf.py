"""Markdown 转 PDF 工具模块

将 Markdown 文件转换为 PDF，支持图表嵌入和中文显示。
依赖: markdown2, weasyprint
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import markdown2
from weasyprint import HTML, CSS
from weasyprint.text.fonts import FontConfiguration


def _find_cjk_font_path() -> str | None:
    """查找系统中可用的 CJK 字体文件路径"""
    candidates = [
        "/usr/share/fonts/google-droid/DroidSansFallback.ttf",
        "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJKsc-Regular.otf",
        "/usr/share/fonts/wqy-microhei/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


_CJK_FONT_PATH = _find_cjk_font_path()

_FONT_FACE_CSS = ""
if _CJK_FONT_PATH:
    _FONT_FACE_CSS = f"""
@font-face {{
    font-family: "CJKFont";
    src: url("file://{_CJK_FONT_PATH}");
}}
"""

CSS_STYLE = _FONT_FACE_CSS + """
@page {
    size: A4;
    margin: 2cm;
}
body {
    font-family: "CJKFont", "Droid Sans Fallback", "Droid Sans",
                 "Noto Sans CJK SC", "WenQuanYi Micro Hei",
                 "Arial", sans-serif;
    font-size: 12px;
    line-height: 1.6;
    color: #333;
}
h1 {
    font-size: 22px;
    color: #1a1a1a;
    border-bottom: 2px solid #1976D2;
    padding-bottom: 8px;
    margin-top: 20px;
}
h2 {
    font-size: 18px;
    color: #1976D2;
    border-bottom: 1px solid #ddd;
    padding-bottom: 6px;
    margin-top: 16px;
}
h3 {
    font-size: 15px;
    color: #333;
    margin-top: 12px;
}
table {
    border-collapse: collapse;
    width: 100%;
    margin: 10px 0;
    font-size: 11px;
}
th, td {
    border: 1px solid #ddd;
    padding: 6px 10px;
    text-align: left;
}
th {
    background-color: #f5f5f5;
    font-weight: bold;
}
tr:nth-child(even) {
    background-color: #fafafa;
}
img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 10px auto;
}
hr {
    border: none;
    border-top: 1px solid #ddd;
    margin: 16px 0;
}
strong {
    color: #1a1a1a;
}
code {
    background-color: #f5f5f5;
    padding: 2px 4px;
    border-radius: 3px;
    font-size: 11px;
}
"""


def convert_md_to_pdf(md_path: str, pdf_path: str) -> str:
    """将 Markdown 文件转换为 PDF 文件

    Args:
        md_path: Markdown 文件路径
        pdf_path: 输出 PDF 文件路径

    Returns:
        生成的 PDF 文件路径
    """
    md_file = Path(md_path).resolve()
    pdf_file = Path(pdf_path)
    pdf_file.parent.mkdir(parents=True, exist_ok=True)

    md_content = md_file.read_text(encoding="utf-8")

    # 将 Markdown 中的相对图片路径转换为绝对路径（file:// 协议）
    md_dir = md_file.parent
    def resolve_image_path(match):
        alt_text = match.group(1)
        img_path = match.group(2)
        if not img_path.startswith(("http://", "https://", "file://")):
            absolute_path = (md_dir / img_path).resolve()
            return f"![{alt_text}](file://{absolute_path})"
        return match.group(0)

    md_content = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", resolve_image_path, md_content)

    html_body = markdown2.markdown(
        md_content,
        extras=["tables", "fenced-code-blocks", "header-ids", "strike"],
    )

    full_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>{CSS_STYLE}</style>
</head>
<body>
{html_body}
</body>
</html>"""

    font_config = FontConfiguration()
    html_doc = HTML(string=full_html)
    css = CSS(string=CSS_STYLE, font_config=font_config)
    html_doc.write_pdf(str(pdf_file), stylesheets=[css], font_config=font_config)
    return str(pdf_file)
