"""
プロンプトファイルローダー

.mdファイルからプロンプトを読み込むためのユーティリティ。
Frontmatter形式でメタデータ（title, description）を含むことができる。
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class PromptFile:
    """プロンプトファイルの内容を表すデータクラス"""

    path: Path
    title: str
    description: str
    content: str

    @property
    def filename(self) -> str:
        """ファイル名（拡張子なし）を返す"""
        return self.path.stem


def expand_data_placeholders(content: str, data_dir: Path) -> str:
    """
    プロンプト内のデータプレースホルダーを展開する。

    {{data:filename}} を対応するファイル内容に置換。

    Args:
        content: プレースホルダーを含むプロンプト本文
        data_dir: データファイルが格納されているディレクトリ

    Returns:
        プレースホルダーが展開されたテキスト
    """
    pattern = r"\{\{data:([^}]+)\}\}"

    def replace_match(match):
        filename = match.group(1)
        data_path = data_dir / filename
        if data_path.exists():
            return data_path.read_text(encoding="utf-8").strip()
        return f"[データファイル未検出: {filename}]"

    return re.sub(pattern, replace_match, content)


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """
    Frontmatter形式のテキストをパースする。

    Args:
        text: Frontmatterを含む可能性のあるテキスト

    Returns:
        (メタデータdict, 本文) のタプル
    """
    # Frontmatterパターン: ---で囲まれた部分
    pattern = r"^---\s*\n(.*?)\n---\s*\n(.*)$"
    match = re.match(pattern, text, re.DOTALL)

    if not match:
        # Frontmatterがない場合
        return {}, text.strip()

    frontmatter_text = match.group(1)
    content = match.group(2).strip()

    # 簡易的なYAMLパース（key: value形式のみ対応）
    metadata = {}
    for line in frontmatter_text.split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip()

    return metadata, content


def load_prompt_file(file_path: Path, data_dir: Optional[Path] = None) -> PromptFile:
    """
    単一のプロンプトファイルを読み込む。

    Args:
        file_path: プロンプトファイルのパス
        data_dir: データファイルのディレクトリ（プレースホルダー展開用）

    Returns:
        PromptFileオブジェクト
    """
    text = file_path.read_text(encoding="utf-8")
    metadata, content = parse_frontmatter(text)

    # データプレースホルダーを展開
    if data_dir is not None and data_dir.exists():
        content = expand_data_placeholders(content, data_dir)

    return PromptFile(
        path=file_path,
        title=metadata.get("title", file_path.stem),
        description=metadata.get("description", ""),
        content=content,
    )


def list_prompt_files(prompts_dir: Path, data_dir: Optional[Path] = None) -> list[PromptFile]:
    """
    指定ディレクトリ内のすべての.mdファイルを読み込む。

    Args:
        prompts_dir: promptsフォルダのパス
        data_dir: データファイルのディレクトリ（プレースホルダー展開用）

    Returns:
        PromptFileオブジェクトのリスト
    """
    if not prompts_dir.exists():
        return []

    prompt_files = []
    for md_file in sorted(prompts_dir.glob("*.md")):
        try:
            prompt_files.append(load_prompt_file(md_file, data_dir))
        except Exception:
            # 読み込みエラーは無視
            pass

    return prompt_files


def get_prompts_for_agent(agent_type: str, base_dir: Optional[Path] = None) -> list[PromptFile]:
    """
    特定のエージェントタイプ用のプロンプト一覧を取得する。

    Args:
        agent_type: "finance" または "medical"
        base_dir: プロジェクトのベースディレクトリ（デフォルトはこのファイルの親の親）

    Returns:
        PromptFileオブジェクトのリスト
    """
    if base_dir is None:
        base_dir = Path(__file__).parent.parent

    prompts_dir = base_dir / agent_type / "prompts"
    data_dir = base_dir / agent_type / "data"
    return list_prompt_files(prompts_dir, data_dir)
