from pathlib import Path

def get_reference_pdfs(data_dir: Path) -> list[str]:
    """
    Retorna a lista de arquivos PDF na pasta informada.

    Args:
        data_dir (Path): Diretório onde estão os PDFs de referência.
    Returns:
        list[str]: Lista de nomes de arquivos PDF.
    """
    return [p.name for p in data_dir.glob('*.pdf')]

def derive_languages_from_filenames(pdf_files: list[str]) -> list[str]:
    """
    Extrai nomes de idiomas a partir dos nomes dos arquivos PDF, assumindo formato '<idioma>_*.pdf'.

    Args:
        pdf_files (list[str]): Lista de nomes de arquivos PDF.
    Returns:
        list[str]: Lista única de idiomas extraídos.
    """
    langs = set()
    for fn in pdf_files:
        lang = fn.split('_')[0]
        langs.add(lang.capitalize())
    return sorted(langs)
