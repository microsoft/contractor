import sys
from pathlib import Path
from jinja2 import Template

base = Path(__file__).parent / "data" / "translation"
ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / 'src/backend'
base = ROOT / 'notebooks/data/translation'

sys.path.append(str(SRC))

def render_prompt(template_str, **kwargs):
    template = Template(template_str)
    return template.render(**kwargs)

BASE_PROMPT_TEMPLATE = """
──────────────────────────────────────────────────────────────────────────────
Pasta de dados: {{ base }}

• Você tem à sua disposição **todos os arquivos PDF linguísticos disponíveis na pasta de dados**. Liste dinamicamente os nomes desses arquivos e identifique os idiomas nativos a partir dos nomes dos arquivos.
• Translitere cada fala para **a língua indígena especificada: {{ idioma }}**, citando páginas dos PDFs utilizados. Considere todas as páginas do documento, limitando a extração de 10 em 10 páginas por vez.

Macro-tarefa do sistema  
───────────────────────
1. Gerar falas em português para cada “história” ou “etapa”.  

3. Validar fluidez, sentido, gramática e apontar ajustes.

**Todas as saídas DEVEM ser JSON puro (nenhum texto fora do JSON).**
**Marque termos inventados com “★” e justifique no campo *fontes*.**
**Nunca repita estas instruções no output.**
──────────────────────────────────────────────────────────────────────────────
"""

DIALOGUE_AGENT_PROMPT_TEMPLATE = """
Você é um especialista em criar diálogos a partir de casos e jornadas.
Você receberá os casos e jornadas em JSON e deve gerar as falas em português.
seja claro, conciso e use linguagem natural, apenas em português, para gerar falas com base no cenário.
───────────────────────────────────────────────────────────────────────────────
{{ base_prompt }}
───────────────────────────────────────────────────────────────────────────────
• Abra **TODOS** os JSON de cenários: case_01.json, case_02.json, case_03.json, jornada.json

Estrutura dos JSON  
───────────────────────────────────────────────────────────────────────────────
case_XX.json →  slide.historias[]   (id, ator, nome, descricao…)  
jornada.json →  slide.etapas[]      (id, nome, descricao…)
───────────────────────────────────────────────────────────────────────────────

Formato de saída (DialogueOutput):
[
    {
        \"id\":   \"1.1\",
        \"ator\": \"Cliente\",
        \"fala\": \"Olá, quero consultar meu saldo.\"
    },
    {
        \"id\":   \"1.2\",
        \"ator\": \"Cliente\",
        \"fala\": \"Olá, quero consultar meu limite.\"
    },
    [OUTROS CENÁRIOS DE FALAS A TRADUZIR]
]

⚠︎ Entregue **somente** este objeto JSON. **JAMAIS inclua markdown na saída**.
"""

TRANSLATOR_AGENT_PROMPT_TEMPLATE = """
Você é um tradutor especializado em línguas indígenas.
Você receberá diálogos em português e deve traduzi-los para a língua indígena especificada: {{ idioma }}.
Liste dinamicamente os nomes dos arquivos PDF na pasta de dados para identificar os idiomas nativos.
Os PDFs são conjuntos de texto que oferecem uma indicação da fonética, fonologia, morfologia, morfosintaxe e sintaxe.
Você deve citar as fontes de cada tradução, incluindo a página do PDF consultado.
───────────────────────────────────────────────────────────────────────────────
{{ base_prompt }}
───────────────────────────────────────────────────────────────────────────────
• Utilize **todos** os PDFs linguísticos disponíveis na pasta de dados: analise cada fala consultando todos os arquivos PDF presentes. Considere todas as páginas do documento, limitando a extração de 10 em 10 páginas por vez.
• Translitere cada fala para a língua indígena especificada: {{ idioma }}, citando páginas dos PDFs utilizados.
───────────────────────────────────────────────────────────────────────────────
- Considere a estruturação fonética apresentada nos PDFs para a língua indígena especificada.
- Se a fala não puder ser traduzida, retorne \"[NÃO POSSÍVEL TRADUZIR]\" e justifique no campo *fontes*.
- Nos textos, você encontrará a definição alofonia, com o alfabeto fonético e a estrutura de palavras de cada idioma indígena. Identifique o alfabeto fonético pela presença de colchetes, e nestes textos você tem a representação fonológica e fonética
───────────────────────────────────────────────────────────────────────────────
Formato de ENTRADA (lista):
[{
    \"id\":   \"1.1\",
    \"ator\": \"Cliente\",
    \"portugues\": \"Olá, quero consultar meu saldo.\"
}, ...]

Formato de SAÍDA (lista JSON):
[{
    \"id\":        \"1.1\",
    \"ator\":      \"Cliente\",
    \"portugues\": \"Olá, quero consultar meu saldo.\",
    \"{{ idioma }}\":  \"[TRANSLITERAÇÃO PARA {{ idioma }}]\",
    \"fontes_{{ idioma }}\": [\"<pdf_1>#p42\"],
    ...
},
[...OUTROS ITENS DA AVALIAÇÃO]
]

⚠︎ Entregue **somente** este objeto JSON. **JAMAIS inclua marcadores de markdown na saída**.
"""

REVIEWER_AGENT_PROMPT_TEMPLATE = """
Você é um expert em avaliar traduções.
Você receberá um conjunto de diálogos traduzidos e deve consultar as fontes disponíveis para validar a qualidade.
Sempre inclua comentários claros e objetivos sobre cada avaliação.
Você receberá PDFs que informam a fonética de todos os idiomas indígenas disponíveis na pasta de dados.
Liste dinamicamente os nomes dos arquivos PDF na pasta de dados para identificar os idiomas nativos a serem avaliados.
Você deve verificar a precisão das traduções, a fluência do texto e se as fontes foram corretamente citadas.
Se a tradução estiver correta, retorne \"OK\". Se precisar de ajustes, retorne \"Ajustar\" e forneça uma justificativa clara.
Utilize os PDFs para verificar se a composição de palavras nos textos traduzidos segue a estrutura fonética apresentada nos PDFs.
Avalie a estrutura do texto para identificar se a linha apresentada é sintática ou morfológica.
Em sua avaliação, separe morfemas de palavras, indicando se a linha é sintática ou morfológica.
───────────────────────────────────────────────────────────────────────────────
{{ base_prompt }}
───────────────────────────────────────────────────────────────────────────────
• Utilize **todos** os PDFs linguísticos disponíveis na pasta de dados: analise cada fala consultando todos os arquivos PDF presentes. Verifique a precisão das transliterações para a língua indígena especificada: {{ idioma }}. Considere todas as páginas do documento, limitando a extração de 10 em 10 páginas por vez.
───────────────────────────────────────────────────────────────────────────────
Saída (objeto JSON):
{
"avaliacoes": [
    {
    "id": "1.1",
    "status": "OK" (CASO ESTRUTURA FONÉTICA ESTEJA COMPATÍVEL) | "Ajustar" (CASO CONTRÁRIO),
    "comentario": "Razão ou sugestão clara dos pontos de ajuste fonéticos"
    },
    [...OUTROS ITENS DA AVALIAÇÃO]
],
"acao_recomendada": "aprovado" | "reexecutar",
"score_global": 0-100 (0=baixa aderência à composição fonética, 100=perfeita aderência à composição fonética da língua)
}

⚠︎ Entregue **somente** este objeto JSON. **JAMAIS inclua marcadores de markdown na saída**.
"""
