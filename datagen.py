from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from colorama import Fore 

import json
from typing import List 
from pydantic import BaseModel
from litellm import completion
from generated_prompt import prompt_template

class Record(BaseModel):
    question: str
    answer: str

class Response(BaseModel):
    generated: List[Record]

def llm_call(data: str, num_records: int = 5) -> List[Record]:
    stream = completion(
        model="ollama_chat/qwen2.5-coder:14b",
        messages=[
            {
                "role": "user",
                "content": prompt_template(data, num_records),
            }
        ],
        stream=True,
        options={"num_predict": 2000},
        format=Response.model_json_schema(),
    )
    raw_output = ""
    for x in stream: 
        delta = x['choices'][0]["delta"]["content"]
        if delta is not None: 
            print(Fore.LIGHTBLUE_EX + delta + Fore.RESET, end="") 
            raw_output += delta 
    return Response.parse_raw(raw_output).generated

if __name__ == "__main__": 
    converter = DocumentConverter()
    doc = converter.convert("networkonfpga.pdf").document
    chunker = HybridChunker()
    chunks = chunker.chunk(dl_doc=doc)

    final_output = []  # flat list of Q&A pairs

    for i, chunk in enumerate(chunks): 
        print(Fore.YELLOW + f"\nRaw Text:\n{chunk.text[:300]}…" + Fore.RESET)
        enriched_text = chunker.contextualize(chunk=chunk)
        print(Fore.LIGHTMAGENTA_EX + f"\nContextualized Text:\n{enriched_text[:300]}…" + Fore.RESET)

        records = llm_call(enriched_text)

        for record in records:
            final_output.append({
                "question": f"{enriched_text.strip()}\n{record.question}",
                "answer": record.answer.strip()
            })

    with open('data.json', 'w', encoding='utf-8') as f: 
        json.dump(final_output, f, indent=2, ensure_ascii=False)
