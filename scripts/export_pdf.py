"""批量将 output/md 下的 Markdown 文件导出为 output/pdf 下对应目录的 PDF 文件

自动扫描 output/md/ 目录下所有 .md 文件，保持完全一致的目录结构，
将其转换为 PDF 并保存到 output/pdf/ 对应路径下。

用法:
    python scripts/export_pdf.py                                        # 转换所有 md 文件
    python scripts/export_pdf.py --dir 20260330_114434                  # 只转换指定时间戳目录
    python scripts/export_pdf.py --file 20260330_114434/analyze_09988.md  # 只转换指定单个文件
    python scripts/export_pdf.py --force                                # 强制覆盖已有 PDF
"""
from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from openquant.utils.md_to_pdf import convert_md_to_pdf


MD_ROOT = Path("output/md")
PDF_ROOT = Path("output/pdf")


def find_md_files(target_dir: str | None = None) -> list[Path]:
    """扫描 output/md 下的所有 .md 文件

    Args:
        target_dir: 可选，指定只扫描某个子目录（如时间戳目录名）

    Returns:
        所有找到的 .md 文件路径列表
    """
    search_root = MD_ROOT / target_dir if target_dir else MD_ROOT
    if not search_root.exists():
        print(f"❌ 目录不存在: {search_root}")
        return []
    return sorted(search_root.rglob("*.md"))


def convert_single(file_path: str, force: bool = False):
    """转换单个 md 文件为 pdf

    Args:
        file_path: 相对于 output/md/ 的文件路径（如 20260330_114434/analyze_09988.md）
        force: 是否强制覆盖已有 PDF
    """
    md_path = MD_ROOT / file_path
    if not md_path.exists():
        # 也尝试作为绝对路径或相对于当前目录的路径
        md_path = Path(file_path)
        if not md_path.exists():
            print(f"❌ 文件不存在: {file_path}")
            print(f"   请提供相对于 {MD_ROOT}/ 的路径，如: 20260330_114434/analyze_09988.md")
            return

    # 确定相对路径
    try:
        relative_path = md_path.relative_to(MD_ROOT)
    except ValueError:
        relative_path = md_path.relative_to(Path.cwd() / MD_ROOT) if (Path.cwd() / MD_ROOT) in md_path.parents else md_path.name
        relative_path = Path(relative_path)

    pdf_relative = relative_path.with_suffix(".pdf")
    pdf_path = PDF_ROOT / pdf_relative

    if pdf_path.exists() and not force:
        print(f"  ⏭️  跳过 (已存在): {pdf_relative}")
        print(f"  使用 --force 强制覆盖")
        return

    print(f"  📑 转换中: {relative_path} → {pdf_relative} ...", end=" ")
    try:
        convert_md_to_pdf(str(md_path), str(pdf_path))
        file_size = pdf_path.stat().st_size
        print(f"✅ ({file_size / 1024:.0f} KB)")
        print(f"\n  📁 PDF 已保存到: {pdf_path}")
    except Exception as exc:
        print(f"❌ 失败: {exc}")


def convert_all(target_dir: str | None = None, force: bool = False):
    """批量转换 md 文件为 pdf

    Args:
        target_dir: 可选，只转换指定子目录
        force: 是否强制覆盖已有 PDF
    """
    md_files = find_md_files(target_dir)
    if not md_files:
        print("⚠️ 未找到任何 .md 文件")
        return

    print(f"📂 扫描到 {len(md_files)} 个 Markdown 文件")
    print(f"   来源: {MD_ROOT}/{'**' if not target_dir else target_dir}")
    print(f"   目标: {PDF_ROOT}/")
    print()

    success_count = 0
    skip_count = 0
    fail_count = 0

    for md_path in md_files:
        relative_path = md_path.relative_to(MD_ROOT)
        pdf_relative = relative_path.with_suffix(".pdf")
        pdf_path = PDF_ROOT / pdf_relative

        if pdf_path.exists() and not force:
            print(f"  ⏭️  跳过 (已存在): {pdf_relative}")
            skip_count += 1
            continue

        print(f"  📑 转换中: {relative_path} → {pdf_relative} ...", end=" ")
        try:
            convert_md_to_pdf(str(md_path), str(pdf_path))
            file_size = pdf_path.stat().st_size
            print(f"✅ ({file_size / 1024:.0f} KB)")
            success_count += 1
        except Exception as exc:
            print(f"❌ 失败: {exc}")
            fail_count += 1

    print(f"\n{'='*60}")
    print(f"  转换完成!")
    print(f"  ✅ 成功: {success_count}")
    if skip_count:
        print(f"  ⏭️  跳过: {skip_count} (使用 --force 强制覆盖)")
    if fail_count:
        print(f"  ❌ 失败: {fail_count}")
    print(f"  📁 PDF 输出目录: {PDF_ROOT}/")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="批量将 output/md 下的 Markdown 文件导出为 PDF"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="只转换指定的子目录 (如时间戳目录名 20260330_114434)",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="只转换指定的单个文件 (相对于 output/md/ 的路径，如 20260330_114434/analyze_09988.md)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制覆盖已有的 PDF 文件",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Markdown → PDF 导出工具")
    print("=" * 60)
    print()

    if args.file:
        convert_single(file_path=args.file, force=args.force)
    else:
        convert_all(target_dir=args.dir, force=args.force)


if __name__ == "__main__":
    main()
