"""
Text chunking utilities.
Splits parsed document records into smaller overlapping chunks while
preserving source/page/section metadata on every chunk.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


def chunk_records(records: list[dict]) -> list[dict]:
    """
    Take parsed document records (one per page/slide/section block) and
    split each one's text into smaller chunks, propagating metadata.

    Returns a list of dicts:
        {
            "text": str,
            "source": str,
            "page": int | None,
            "section": str | None,
            "chunk_id": str,   # unique id, e.g. "report.pdf_p3_c0"
        }
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for record in records:
        pieces = splitter.split_text(record["text"])
        for i, piece in enumerate(pieces):
            page_label = record.get("page")
            loc_tag = f"p{page_label}" if page_label is not None else "sec"
            chunks.append({
                "text": piece,
                "source": record["source"],
                "page": record.get("page"),
                "section": record.get("section"),
                "chunk_id": f"{record['source']}_{loc_tag}_c{i}",
            })
    return chunks
