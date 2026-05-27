"""RAG VDB Exercise 2: import and structure NACE documents."""

from dataclasses import dataclass, field
from typing import Optional

import duckdb

PATH_NACE = "https://minio.lab.sspcloud.fr/projet-formation/diffusion/funathon/2026/project2/NACE_Rev2.1_Structure_Explanatory_Notes_EN.tsv"


def _load_httpfs(con: duckdb.DuckDBPyConnection) -> None:
    try:
        con.execute("LOAD httpfs;")
    except duckdb.Error:
        con.execute("INSTALL httpfs;")
        con.execute("LOAD httpfs;")


def _clean(value) -> Optional[str]:
    """Normalize to stripped single-line string, or None if empty/missing."""
    if value is None:
        return None
    cleaned = " ".join(str(value).replace("\n", " ").split())
    return cleaned or None


@dataclass
class NaceDocument:
    code: str
    heading: str
    level: int
    parent_code: Optional[str] = None
    includes: Optional[str] = None
    includes_also: Optional[str] = None
    excludes: Optional[str] = None

    text: str = field(init=False)

    @classmethod
    def from_raw(
        cls,
        raw: dict,
        with_includes_also: bool = True,
        with_excludes: bool = False,
    ) -> "NaceDocument":
        for key in ("CODE", "HEADING", "LEVEL"):
            if not raw.get(key):
                raise ValueError(f"Missing required field: {key}")

        level = int(raw["LEVEL"])
        if not (1 <= level <= 4):
            raise ValueError(f"Invalid level: {level}")

        parent_code = _clean(raw.get("PARENT_CODE"))
        if level > 1 and not parent_code:
            raise ValueError(f"Missing parent code for hierarchical level {level}: {raw['CODE']}")

        obj = cls(
            code=str(raw["CODE"]).strip(),
            heading=_clean(raw["HEADING"]),
            level=level,
            parent_code=parent_code,
            includes=_clean(raw.get("Includes")),
            includes_also=_clean(raw.get("IncludesAlso")),
            excludes=_clean(raw.get("Excludes")),
        )

        obj.text = obj.to_embedding_text(
            with_includes_also=with_includes_also,
            with_excludes=with_excludes,
        )
        return obj

    def to_embedding_text(
        self,
        *,
        with_includes_also: bool = False,
        with_excludes: bool = False,
    ) -> str:
        parts = [
            f"# Code: {self.code}",
            f"# Title: {self.heading}",
        ]

        if self.includes:
            parts.extend(["", "## Includes:", self.includes.strip()])
        if with_includes_also and self.includes_also:
            parts.extend(["", "## Also includes:", self.includes_also.strip()])
        if with_excludes and self.excludes:
            parts.extend(["", "## Excludes:", self.excludes.strip()])

        output = "\n".join(parts).replace("\\n", "\n")
        return output.strip()


def main() -> None:
    con = duckdb.connect(database=":memory:")
    _load_httpfs(con)

    table = con.execute(
        f"""
        SELECT *
        FROM read_csv(
            '{PATH_NACE}',
            delim='\t',
            header=true,
            all_varchar=true
        )
        """
    ).to_arrow_table()
    nace = table.to_pylist()

    print("Loaded rows:", len(nace))
    print("Sample raw record (index 22):")
    print(nace[22])

    nace_documents = [
        NaceDocument.from_raw(raw=row, with_includes_also=True, with_excludes=True)
        for row in nace
    ]
    print("\nBuilt NaceDocument count:", len(nace_documents))

    i = 50
    doc = NaceDocument.from_raw(nace[i], with_includes_also=True, with_excludes=True)
    print("\n=== WITH exclusions ===")
    print(doc.text)

    print("\n=== WITHOUT exclusions ===")
    print(doc.to_embedding_text(with_includes_also=True, with_excludes=False))


if __name__ == "__main__":
    main()
