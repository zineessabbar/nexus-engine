import re
import fitz
import docx
import easyocr
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from logger import get_logger

logger = get_logger("document_loader")


def load_txt(path):
    logger.info(f"Chargement TXT : {path.name}")
    return path.read_text(encoding="utf-8"), []


def load_docx(path):
    logger.info(f"Chargement DOCX : {path.name}")
    doc = docx.Document(path)
    
    paragraphs = []
    tables_as_text = []

    for element in doc.element.body:
        tag = element.tag.split("}")[-1]
        
        if tag == "p":
            para = docx.text.paragraph.Paragraph(element, doc)
            if para.text.strip():
                paragraphs.append(para.text.strip())
        
        elif tag == "tbl":
            table = docx.table.Table(element, doc)
            table_text = _table_to_text(table)
            if table_text:
                tables_as_text.append(table_text)
                paragraphs.append(table_text)

    return "\n".join(paragraphs), tables_as_text


def load_pdf(path):
    logger.info(f"Chargement PDF : {path.name}")
    doc = fitz.open(path)
    
    full_text = []
    tables_as_text = []
    ocr_reader = None

    for page_num, page in enumerate(doc):
        text = page.get_text().strip()
        
        if not text or len(text) < 50:
            logger.warning(f"Page {page_num + 1}— OCR requis")
            if ocr_reader is None:
                ocr_reader = easyocr.Reader(["fr", "en"], gpu=True)
            text = _ocr_page(page, ocr_reader)
        
        if text:
            full_text.append(text)
        
        # Détection des tableaux via blocs de texte
        tables = _extract_pdf_tables(page)
        tables_as_text.extend(tables)

    if ocr_reader is not None:
        logger.info("OCR appliqué avec succès.")

    return "\n".join(full_text), tables_as_text


def _ocr_page(page, ocr_reader):
    try:
        pix = page.get_pixmap(dpi=300)
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        results = ocr_reader.readtext(img_array)
        return " ".join([text for _, text, conf in results if conf > 0.5])
    except Exception as e:
        logger.error(f"Erreur OCR : {e}")
        return ""


def _extract_pdf_tables(page):
    tables = []
    try:
        tab = page.find_tables()
        for table in tab.tables:
            rows = table.extract()
            if rows:
                table_text = _rows_to_text(rows)
                if table_text:
                    tables.append(table_text)
    except Exception:
        pass
    return tables


def _table_to_text(table):
    if not table.rows:
        return ""
    
    headers = [cell.text.strip() for cell in table.rows[0].cells]
    headers = [h for h in headers if h]
    
    if not headers:
        return ""
    
    lines = []
    for row in table.rows[1:]:
        cells = [cell.text.strip() for cell in row.cells]
        if any(cells):
            line = " | ".join(
                f"{headers[i]}: {cells[i]}"
                for i in range(min(len(headers), len(cells)))
                if cells[i]
            )
            if line:
                lines.append(line)
    
    return "\n".join(lines)


def _rows_to_text(rows):
    if not rows or not rows[0]:
        return ""
    
    headers = [str(cell).strip() if cell else "" for cell in rows[0]]
    headers = [h for h in headers if h]
    
    if not headers:
        return ""
    
    lines = []
    for row in rows[1:]:
        if not row:
            continue
        cells = [str(cell).strip() if cell else "" for cell in row]
        if any(cells):
            line = " | ".join(
                f"{headers[i]}: {cells[i]}"
                for i in range(min(len(headers), len(cells)))
                if i < len(cells) and cells[i]
            )
            if line:
                lines.append(line)
    
    return "\n".join(lines)

def clean_extracted_text(text):
    import re
    
    # Réassemble les mots coupés : "Techno-\nlogy" → "Technology"
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    
    # Supprime les sauts de ligne simples à l'intérieur d'un paragraphe
    # (garde les doubles sauts de ligne qui marquent les vrais paragraphes)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    
    # Normalise les espaces multiples
    text = re.sub(r" {2,}", " ", text)
    
    # Normalise les sauts de ligne multiples (max 2)
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    return text.strip()



def detect_structure(text):
    """Détecte la stratégie de chunking à appliquer."""
    
    if re.search(r"Article\s+\d+(?:\.\d+)?", text):
        logger.debug("Structure détectée : article")
        return "article"
    
    if re.search(r"^\s*\d+\.\d*\s+[A-ZÀÉÈÊÎÔÙ]", text, re.MULTILINE):
        logger.debug("Structure détectée : section")
        return "section"
    
    logger.debug("Structure détectée : semantic")
    return "semantic"




def chunk_by_article(text, doc_source):
    pattern = r"Article\s*(\d+(?:\.\d+)?)\s*-\s*([^\n]+)\n((?:(?!Article\s*\d+(?:\.\d+)?\s*-).*\n?)*)"
    matches = re.findall(pattern, text)
    
    chunks = []
    for number, title, body in matches:
        body = body.strip()
        if not body:
            continue
        chunks.append({
            "article_number": number,
            "title": title.strip(),
            "text": f"Article {number} - {title.strip()}\n{body}".strip(),
            "chunk_type": "article",
            "doc_source": doc_source,
        })
    return chunks


def chunk_by_section(text, doc_source):
    pattern = r"(\d+(?:\.\d+)?)\.\s+([^\n]+)\n((?:(?!\d+(?:\.\d+)?\.\s).*\n?)*)"
    matches = re.findall(pattern, text)
    
    chunks = []
    for number, title, body in matches:
        body = body.strip()
        if not body:
            continue
        chunks.append({
            "article_number": number,
            "title": title.strip(),
            "text": f"{number}. {title.strip()}\n{body}".strip(),
            "chunk_type": "section",
            "doc_source": doc_source,
        })
    return chunks


def chunk_semantic(text, doc_source, model=None):

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 20]
    
    if not sentences:
        return []
    
    if model is None or len(sentences) < 3:
        # Fallback : chunks de taille fixe si pas de modèle ou trop peu de phrases
        return _chunk_fixed_size(sentences, doc_source)
    
    embeddings = model.encode(sentences, normalize_embeddings=True)
    similarities = [float(np.dot(embeddings[i-1],embeddings[i]))for i in range(1,len(embeddings))]

    mean_sim=np.mean(similarities)
    std_sim=np.std(similarities)
    dynamic_threshold=(mean_sim -std_sim)
    
    logger.debug(f"[{doc_source}] Seuil sémentique dynamique calculé:{dynamic_threshold:.3f} (Moyenne: {mean_sim:.3f})")
    breakpoints = []
    for i,sim in enumerate(similarities):
        if sim < dynamic_threshold:
            breakpoints.append(i+1)
    
    chunks_text = []
    prev = 0
    for bp in breakpoints:
        chunk = " ".join(sentences[prev:bp])
        if chunk:
            chunks_text.append(chunk)
        prev = bp
    chunks_text.append(" ".join(sentences[prev:]))
    
    return [
        {
            "article_number": str(i + 1),
            "title": f"Section {i + 1}",
            "text": chunk,
            "chunk_type": "semantic",
            "doc_source": doc_source,
        }
        for i, chunk in enumerate(chunks_text)
        if chunk.strip()
    ]


def _chunk_fixed_size(sentences, doc_source, size=5):
    chunks = []
    for i in range(0, len(sentences), size):
        group = sentences[i:i + size]
        chunks.append({
            "article_number": str(i // size + 1),
            "title": f"Bloc {i // size + 1}",
            "text": " ".join(group),
            "chunk_type": "fixed",
            "doc_source": doc_source,
        })
    return chunks


def chunk_tables(tables, doc_source,max_lines=15):
    chunks = []
    for table_idx, table_text in enumerate(tables):
        lines=[line.strip() for line in table_text.split("\n") if line.strip()]
        if not lines:
            continue

        for i in range(0,len(lines),max_lines):
            group=lines[i:i+max_lines]
            part_number=(i//max_lines)+1

            title=f"Table {table_idx+1}"
            if len(lines) > max_lines:
                title=f"Partie {part_number}"
            
            chunks.append({
                "article_number": f"{table_idx+1}.{part_number}",
                "title": title,
                "text": "\n".join(group),
                "chunk_type": "table",
                "doc_source": doc_source,
            })
        
    return chunks




def load_and_chunk(path, embedding_model=None):

    path = Path(path)
    doc_source = path.name
    
    logger.info(f"Traitement de '{doc_source}'")
    
    # Extraction
    suffix = path.suffix.lower()
    if suffix == ".txt":
        text, tables = load_txt(path)
    elif suffix == ".pdf":
        text, tables = load_pdf(path)
    elif suffix in (".docx", ".doc"):
        text, tables = load_docx(path)
    else:
        raise ValueError(f"Format non supporté : {suffix}")

    # Nettoyage du texte extrait (césures, espaces, sauts de ligne)
    if suffix != ".txt":  # le TXT n'a pas besoin de nettoyage
        text = clean_extracted_text(text)
    
    # Détection de structure
    strategy = detect_structure(text)
    logger.info(f"Stratégie de chunking : {strategy}")
    
    # Chunking
    if strategy == "article":
        chunks = chunk_by_article(text, doc_source)
    elif strategy == "section":
        chunks = chunk_by_section(text, doc_source)
    else:
        chunks = chunk_semantic(text, doc_source, model=embedding_model)
    
    # Tableaux en chunks séparés
    if tables:
        logger.info(f"{len(tables)} tableau(x) détecté(s) — chunking tabulaire")
        chunks += chunk_tables(tables, doc_source)
    
    logger.info(f"'{doc_source}' → {len(chunks)} chunks générés (stratégie: {strategy})")
    return chunks