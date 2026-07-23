import re 
def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
def chunk_text(text,chunk_size,overlap) ->list[str]:
    normalize_text = clean_text(text)
    words = normalize_text.split(normalize_text,' ')
    end = 0
    start = 0
    chunk = None
    chunk_array=[]
    chunk_word=None
    for word in words:
       
        if len(" ".join (chunk_word,word))>=chunk_size:
            chunk_array = chunk_array.append(chunk_word)


    while end <= len(words):
        if end== 0:
            end = chunk_size
        chunk = normalize_text[start:end]
        if end >= len(normalize_text):
            chunk_array.append(chunk)
        start = start + chunk_size - overlap  
        end = start  +   chunk_size
        chunk_array.append(chunk)
        
    return chunk_array
print(chunk_text("This is a test for checkinh Chunk",6,3))

