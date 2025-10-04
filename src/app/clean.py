from bs4 import BeautifulSoup
import re
import unicodedata

DENYLIST = [
    "Donasi sekarang", "Bagikan", "Share", "Klik di sini"
]

def html_to_canonical(html: str) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")

    # remove script/style
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # extract headings, paragraphs, lis
    parts = []
    for sel in soup.select("h1,h2,h3,p,li"):
        txt = sel.get_text(separator=" ", strip=True)
        if not txt:
            continue
        if any(d.lower() in txt.lower() for d in DENYLIST):
            continue
        parts.append(txt)

    text = "\n".join(parts)
    # normalize unicode & whitespace
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text

def template_text(title: str, category_name: str, canonical: str) -> str:
    # Simple template; keep it short but structured
    lines = [f"Title: {title}"]
    if category_name:
        lines.append(f"Category: {category_name}")
    if canonical:
        lines.append("Details:")
        lines.append(canonical)
    return "\n".join(lines)
