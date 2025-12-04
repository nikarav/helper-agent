import re

from bs4 import BeautifulSoup


def clean_html(text: str) -> str:
    """
    Remove HTML/MDX tags, keep text content.

    :param text: Text with potential HTML/MDX
    :return: Clean text
    """
    fences = {}

    def save_fence(m: re.Match) -> str:
        key = f"__FENCE_{len(fences)}__"
        fences[key] = m.group(0)
        return key

    text = re.sub(r"```[\s\S]*?```", save_fence, text)

    soup = BeautifulSoup(text, "html.parser")

    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    text = "\n".join(line.strip() for line in text.split("\n") if line.strip())

    # restore code fences
    for key, val in fences.items():
        text = text.replace(key, val)

    return text


def clean_document(content: str, source_file: str) -> str:
    """
    Clean document content based on its source.

    :param content: Document content
    :param source_file: Source identifier ('langchain' or 'langgraph')
    :return: Cleaned content
    """
    if source_file == "langchain":
        return clean_html(content)

    # LangGraph docs are already clean markdown
    return content
