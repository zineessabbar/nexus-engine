import re
from pathlib import Path
#load document
def load_document(path):
    return Path(path).read_text(encoding="utf-8")

# chunk document
def chunk_by_article(text):
    pattern = r"Article\s*(\d+(?:\.\d+)?)\s*-\s*([^\n]+)\n(.*?)(?=\n\s*Article|\Z)"
    matches = re.findall(pattern,text)
    
    chunks=[]
    for number,title,body in matches:
        body=body.strip()
        chunks.append(
            {
                "article_number":number,
                "title":title.strip(),
                "text":f"Article {number} - {title.strip()} \n {body}".strip(),
            }
        )
    return chunks


def main():
    text=load_document("./documents/normes.txt")
    chunks=chunk_by_article(text)

    for i in chunks[:3]:
        print(i)

if __name__ == "__main__":
    main()