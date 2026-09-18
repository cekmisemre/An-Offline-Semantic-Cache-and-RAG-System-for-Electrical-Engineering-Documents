import os
import re
import csv
import sys
import json
import time
import sqlite3
import hashlib
import urllib.error
import urllib.request
import numpy as np
from openai import OpenAI

MAX_CACHE_RECORDS = 1000  # LRU eviction ceiling

# Measurement support: ask() and answer_with_rag() write their per-call timings here
# instead of only printing them, so a benchmark harness can collect them afterwards.
# A plain module-level dict is used deliberately - it keeps every existing function
# signature and return value unchanged, so no caller breaks.
LAST_TIMINGS = {}

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="no-key-required")

# Dedicated embedding model server - a second llama-server instance (see setup notes)
embedding_client = OpenAI(base_url="http://127.0.0.1:8001/v1", api_key="no-key-required")

# --- Domain focus: the project is now scoped to a single domain ---
# Instead of the earlier tests' generic/mixed topics (physics + Turkish Republic
# trivia), the system now operates in a single specialty domain: oversea education
# advising. This covers the LLM's system prompt, the RAG document base, and the
# cache test scenarios (below, inside __main__).
DOMAIN_NAME = "electrical engineering technical support"

# Single switch for calibration/debug output (best_match_score + entity guard
# details). Set to False for a clean output when demoing/submitting.
VERBOSE_DIAGNOSTICS = False

DOMAIN_SYSTEM_PROMPT = (
    "You are a helpful assistant specialized ONLY in electrical engineering technical "
    "knowledge retrieval - helping engineers and technicians understand electrical "
    "standards, technical manuals, and regulations such as wiring design, overcurrent "
    "protection, grounding, and equipment installation requirements. Answer strictly "
    "using the information provided to you; if the provided information doesn't cover "
    "the question, say so rather than guessing. If a question falls outside this domain, "
    "say briefly that it's outside your area of expertise and do not attempt to answer "
    "it. Answer accurately and exclusively in ENGLISH. Keep answers professional and "
    "concise."
)


def get_llm_response(prompt, system_prompt=DOMAIN_SYSTEM_PROMPT, max_tokens=500):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content


def canonicalize_query(query):
    """Fast, deterministic LLM call that normalizes a question into a
    standard search-query form before it touches the cache, so paraphrases collapse
    onto (nearly) the same canonical string/embedding. (We also tried asking the model
    to normalize named entities here, but the local model didn't follow that reliably -
    see ENTITY_ALIASES / extract_entities() for a deterministic fix instead.)"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": f"Convert this question to a standard search query format: '{query}'"}
        ],
        max_tokens=20,
        temperature=0
    )
    return response.choices[0].message.content.strip()


def get_real_embedding(text):
    """Real semantic embedding from a dedicated llama-server embedding model (e.g. all-MiniLM-L6-v2 GGUF)."""
    response = embedding_client.embeddings.create(model="minilm", input=text)
    return np.array(response.data[0].embedding, dtype=np.float32)


def get_universal_embedding(text):
    # Kept for reference/comparison only - no longer called in the main pipeline below.
    text = text.lower().replace("?", "").replace(".", "").replace(",", "")
    words = text.split()
    fillers = {"how", "to", "can", "you", "find", "best", "the", "a", "for", "is", "where", "make", "an"}
    keywords = [w for w in words if w not in fillers]
    v = np.zeros(384, dtype=np.float32)
    if not keywords:
        keywords = words
    for w in keywords:
        char_sum = sum(ord(c) * (i + 1) for i, c in enumerate(w))
        idx = char_sum % 384
        v[idx] += 2.0
        v[(idx + 1) % 384] += 0.5
        v[(idx - 1) % 384] += 0.5
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


# Helper sets for extract_entities() (see its docstring below)
_QUESTION_STARTERS = {
    "what", "how", "who", "where", "when", "why", "which", "can", "could",
    "is", "are", "do", "does", "did", "tell", "please", "would", "will", "should",
}
_NON_ENTITY_WORDS = {
    "i",
    # Generic English function words. These are never entities in any domain, but they
    # DO get capitalized at the start of a sentence, and the sentence-initial skip above
    # only covers known question-starters - so a query beginning "For a 15-ampere..." or
    # "By what factor..." used to register "for"/"by" as its entity. Two such queries
    # then had disjoint entity sets and the guard rejected a correct semantic match
    # (found while benchmarking the EE query set).
    "a", "an", "the", "and", "or", "of", "for", "by", "in", "on", "at", "to",
    "with", "from", "if", "as", "per", "about", "into", "over", "under",
}

# Known demonym/adjective -> country-name mappings. The LLM-based canonicalizer
# doesn't do this reliably (see project notes), so we solve it with a
# deterministic, extensible dictionary instead.
ENTITY_ALIASES = {
    "turkish": "turkey",
    "bulgarian": "bulgaria",
    "french": "france",
    "german": "germany",
    "italian": "italy",
    "spanish": "spain",
    "american": "america",
    "british": "britain",
    "russian": "russia",
    "chinese": "china",
    "japanese": "japan",
    # Country abbreviations/adjectives common in oversea-education queries
    "us": "america",
    "usa": "america",
    "uk": "britain",
    "canadian": "canada",
    "australian": "australia",
}


def extract_entities(text):
    """Lightweight proper-noun heuristic: capitalized words are treated as entities,
    with two exclusions learned from testing:
    1) sentence-initial words are only skipped if they're a KNOWN question-starter
       (What/How/Tell/...) - not blanket-skipped, so a canonical query that happens to
       START with the entity itself (e.g. "US student visa application...") doesn't lose
       it and silently empty out the entity set (which would by-pass the guard entirely).
    2) the capitalized pronoun "I" is excluded - it's grammatically always capitalized
       mid-sentence but never a distinguishing entity, so keeping it just dilutes the
       overlap score between genuinely-matching questions.
    Known demonym/adjective forms are mapped to their canonical entity name via
    ENTITY_ALIASES, since the LLM canonicalizer doesn't do this reliably."""
    words = re.findall(r"[A-Za-z']+", text)
    entities = set()
    for i, w in enumerate(words):
        # Strip a trailing contraction/possessive ("What's" -> "what", "Turkey's" ->
        # "turkey") so those forms match the sets below instead of slipping through as
        # their own distinct "entity".
        wl = re.sub(r"'\w*$", "", w.lower())
        if not wl:
            continue
        # Single letters are never entities. In EE text the word regex pulls the unit
        # suffix out of tokens like "15A" or "20V" as a lone capital letter, which the
        # capitalization heuristic below would otherwise accept.
        if len(wl) == 1:
            continue
        if i == 0 and wl in _QUESTION_STARTERS:
            continue
        if wl in _NON_ENTITY_WORDS:
            continue
        if w[0].isupper():
            entities.add(ENTITY_ALIASES.get(wl, wl))
    return entities


def entities_agree(query_text, cached_text, min_overlap=0.5):
    """Guards against high cosine similarity between structurally similar but factually
    different questions (e.g. same template, different named entity - 'Turkish Republic'
    vs 'Bulgarian Republic'). If neither text has capitalized entities, defer to embeddings."""
    q_entities = extract_entities(query_text)
    c_entities = extract_entities(cached_text)
    if not q_entities or not c_entities:
        return True
    overlap = len(q_entities & c_entities) / len(q_entities | c_entities)
    return overlap >= min_overlap


# Found while testing pre-warm. entities_agree() catches proper nouns like countries,
# but couldn't tell apart SAME-country, DIFFERENT-TOPIC questions (e.g. "average
# tuition fee" vs "scholarship opportunities") - both matched at 70-76% cosine
# similarity, and since the country was the same, the entity guard let them through
# too. Result: for every country, scholarship questions were incorrectly returning
# the tuition-fee answer.
#
# Fix: a SEPARATE "topic" guard, based on a fixed domain-topic vocabulary. We do NOT
# mix this into the set entities_agree() uses - because a template's fixed words
# (e.g. "visa"/"documents") appear identically in every instance of that template; if
# merged with the country set, they could mask a genuine mismatch between different
# countries and cause a false HIT (caught this exact case while testing the
# Canada/US visa trap). Hence two separate gates: entities_agree() (country/entity)
# AND topics_agree() (topic) - both must pass for a HIT.
TOPIC_KEYWORDS = {
    "ielts", "toefl", "yos", "visa", "document", "documents", "tuition", "fee", "fees",
    "scholarship", "scholarships", "gpa", "transcript", "deadline", "deadlines",
    "processing", "insurance", "work",
    # Added for the week-2 (Task 3) expansion - all 435 pairs were tested and
    # verified (see project notes). The word "application" was deliberately NOT
    # added: it appears in too many templates (gpa_requirement, application_deadline,
    # visa_documents, student_visa_financial_proof, application_fee) and caused false-
    # positive overlap (in particular, gpa_requirement/application_deadline shared two
    # words together with "master's" and hit the 0.5 threshold exactly). A narrower
    # word, "university" (singular), is used instead.
    "cost", "living", "financial", "proof", "interview", "acceptance", "waiver",
    "university", "accommodation", "residency", "pathway", "bank", "account",
    "discounts", "benefits", "internship", "duration", "master's", "bachelor's",
    "credits", "transfer", "intake", "admission", "conditional", "campus",
    "dependents", "employment", "graduates", "rules",
}


def extract_topic_keywords(text):
    """Checks whether words from a fixed domain-topic vocabulary appear in the text
    (case-insensitive). Unlike extract_entities(), this doesn't look at capitalization -
    these words aren't proper nouns, they're fixed domain terms."""
    words = re.findall(r"[a-z']+", text.lower())
    return {w for w in words if w in TOPIC_KEYWORDS}


def topics_agree(query_text, cached_text, min_overlap=0.5):
    """Same logic as entities_agree(), but over topic keywords. If neither side
    contains a known domain term (i.e. none of the vocabulary words appear in either
    text), we defer to the embedding (return True) - just like entities_agree()."""
    q_topics = extract_topic_keywords(query_text)
    c_topics = extract_topic_keywords(cached_text)
    if not q_topics or not c_topics:
        return True
    overlap = len(q_topics & c_topics) / len(q_topics | c_topics)
    return overlap >= min_overlap


# Found while benchmarking the EE domain. entities_agree() catches proper nouns and
# topics_agree() catches subject areas, but in EE the thing that most often separates
# two otherwise near-identical questions is a NUMBER: "a 15 A receptacle" vs "a 30 A
# receptacle" have different correct answers (12 A vs 24 A per Table S-4), and "a 7 kW
# motor" vs "a 3 kW motor" need different breaker curves (Type D vs Type C). This is
# the EE equivalent of the country guard - same failure shape, different domain.
# Confirmed on a real run: "max load on a 30A receptacle" matched the cached 15 A
# question at 58% and returned the 15 A answer. Notably it MISSED on the fast path and
# only matched after canonicalize_query() reworded it, so canonicalization can itself
# blur the numeric detail - which is exactly why a separate deterministic gate is safer
# than trusting either the embedding or the canonicalizer with it.
_UNIT_ALIASES = {
    "a": "a", "amp": "a", "amps": "a", "ampere": "a", "amperes": "a",
    "v": "v", "volt": "v", "volts": "v", "kv": "kv",
    "kw": "kw", "kilowatt": "kw", "kilowatts": "kw",
    "w": "w", "watt": "w", "watts": "w", "kva": "kva",
    "hz": "hz", "hertz": "hz", "%": "%",
    "mm": "mm", "cm": "cm", "m": "m", "meter": "m", "meters": "m",
    "ft": "ft", "foot": "ft", "feet": "ft",
    "in": "in", "inch": "in", "inches": "in",
}

# A number followed by a unit, tolerating "15A", "15 A" and "15-ampere".
_SPEC_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*-?\s*(%|[A-Za-z]+)")


def extract_numeric_specs(text):
    """Returns the set of value+unit specifications in text, normalized (so "15A",
    "15 amperes" and "15-ampere" all collapse to "15a"). Deliberately conservative:
    a number is only kept when it is followed by a KNOWN unit, so section numbers
    ("Section 4.3.2"), regulation numbers ("1910.304") and bare counts ("two or more
    receptacles") are ignored rather than treated as specifications."""
    specs = set()
    for value, unit in _SPEC_PATTERN.findall(text):
        unit_key = _UNIT_ALIASES.get(unit.lower())
        if unit_key is None:
            continue
        specs.add(f"{float(value):g}{unit_key}")
    return specs


def specs_agree(query_text, cached_text):
    """Third gate, alongside entities_agree() and topics_agree(). Unlike those two,
    this does NOT use a Jaccard overlap ratio - it requires a non-empty intersection
    instead. A cached question often legitimately spans several values (the 15 A
    question also mentions the 20 A circuit it sits on), so demanding proportional
    overlap would reject correct paraphrases; what matters is whether the value being
    ASKED about is among the values the cached answer covers. If either side has no
    numeric specification at all, we defer to the embedding, like the other guards."""
    q_specs = extract_numeric_specs(query_text)
    c_specs = extract_numeric_specs(cached_text)
    if not q_specs or not c_specs:
        return True
    return bool(q_specs & c_specs)


# Chunk sizing is measured in TOKENS, not words. all-MiniLM-L6-v2 accepts 256 tokens
# and silently truncates anything longer, and the embedding server rejects a batch over
# its physical batch size outright - which is how this surfaced: a 500-word chunk of
# OSHA text became 802 tokens and crashed ingestion. Words are a poor proxy here.
# Measured on that text: 1.6 tokens per word, against roughly 1.3 for ordinary prose,
# because technical wording is long (6.8 characters per word) and citations like
# "1910.304(g)(6)(i)(B)" split into many tokens.
EMBEDDING_MAX_TOKENS = 256   # the model's input window
CHUNK_TARGET_TOKENS = 175    # aim below it, so overlap and wording variation still fit
# 175 is not arbitrary. The hand-tuned 120-word chunks that produced the best measured
# accuracy so far (74.7%) came to roughly 175-190 tokens on these documents. A first
# attempt at 200 tokens produced 128-138 word chunks and scored 68.9%, so the budget is
# set to reproduce the configuration that was actually measured to work, while still
# being derived per document instead of hand-set.

_TOKENIZE_URL = "http://127.0.0.1:8001/tokenize"
_tokenizer_usable = None  # None = not probed yet, True/False afterwards


def count_tokens(text):
    """Number of tokens in text, from the embedding server's /tokenize endpoint.

    Falls back to a character estimate if that endpoint is missing or unreachable, so
    ingestion still works rather than failing. The fallback deliberately OVER-estimates
    (3.5 characters per token, against the 4.65 measured on the OSHA text) because a
    chunk that is too small is harmless, while one that is too large is either rejected
    by the server or silently truncated."""
    global _tokenizer_usable
    if _tokenizer_usable is not False:
        try:
            request = urllib.request.Request(
                _TOKENIZE_URL,
                data=json.dumps({"content": text}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                tokens = json.loads(response.read()).get("tokens")
            if isinstance(tokens, list):
                _tokenizer_usable = True
                return len(tokens)
            _tokenizer_usable = False
        except (urllib.error.URLError, OSError, ValueError, TimeoutError):
            _tokenizer_usable = False
            print("[TOKENIZER] /tokenize unavailable - using a character-based estimate instead.")
    return int(len(text) / 3.5) + 1


def pick_chunk_size(text, target_tokens=CHUNK_TARGET_TOKENS, sample_words=300):
    """Chooses a word-count chunk size for THIS document, by measuring how many tokens
    its own wording costs per word. This replaces the hand-tuned chunk_size=120 that was
    found by trial and error on one file and would not transfer to a new document."""
    words = text.split()
    if not words:
        return 500
    sample_words_list = words[:min(sample_words, len(words))]
    sample = " ".join(sample_words_list)
    tokens_per_word = count_tokens(sample) / len(sample_words_list)
    return max(int(target_tokens / max(tokens_per_word, 0.1)), 20)


def chunk_text(text, chunk_size=500, overlap=50):
    """
    Splits text into overlapping chunks using a word-based sliding window.
    No external dependencies (no nltk/langchain/tiktoken) - just str.split() and a loop.
    chunk_size and overlap are both measured in words.
    """
    words = text.split()
    if not words:
        return []

    step = max(chunk_size - overlap, 1)  # guards against overlap >= chunk_size (avoids an infinite loop)
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += step

    return chunks


def split_into_blocks(text):
    """Splits text into ("table", block) and ("prose", block) pieces.

    A table is a line that begins with "Table" and ends with a colon, plus the
    consecutive non-empty lines that follow it. This matches how the corpus is written
    (paragraphs and tables separated by blank lines) and does NOT depend on the row
    separator, so it survives small formatting differences. A document produced by a
    different converter may need a different detector - the format the corpus uses is
    the one this understands."""
    blocks = []
    current_kind, current_lines = "prose", []

    def flush():
        if current_lines:
            body = "\n".join(current_lines).strip()
            if body:
                blocks.append((current_kind, body))

    lines = text.split("\n")
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        is_table_heading = stripped.lower().startswith("table") and stripped.endswith(":")
        if is_table_heading:
            flush()
            table_lines = [stripped]
            i += 1
            while i < len(lines) and lines[i].strip():
                table_lines.append(lines[i].strip())
                i += 1
            blocks.append(("table", "\n".join(table_lines)))
            current_kind, current_lines = "prose", []
            continue
        current_lines.append(lines[i])
        i += 1
    flush()
    return blocks


def chunk_table_block(block, chunk_size):
    """Chunks one table, keeping its heading at the top of every chunk it produces.

    A bare table row ("9001 V-25 kV -> Condition A: 1.5 m ...") says nothing about what
    the numbers measure, so on its own it embeds poorly and ranks low for a question
    about working space. Repeating the heading gives every chunk the words the question
    actually uses. Splitting by rows also keeps the table out of the surrounding prose:
    the row Q14 needs used to sit in a chunk that opened with an unrelated sentence
    about lighting, which diluted it."""
    lines = block.split("\n")
    heading, rows = lines[0], [r for r in lines[1:] if r.strip()]
    if not rows:
        return [heading]

    heading_words = len(heading.split())
    chunks, current, current_words = [], [], 0
    for row in rows:
        row_words = len(row.split())
        if current and heading_words + current_words + row_words > chunk_size:
            chunks.append(heading + "\n" + "\n".join(current))
            current, current_words = [], 0
        current.append(row)
        current_words += row_words
    if current:
        chunks.append(heading + "\n" + "\n".join(current))
    return chunks


def chunk_document(text, chunk_size=500, overlap=50):
    """Chunks a whole document, isolating tables from prose.

    Prose is chunked exactly as before, by the same word-based sliding window in
    chunk_text() - that part is unchanged and still measured by the CHUNK CHECK test.
    The only difference is that the window never runs across a table, and each table is
    chunked on its own by chunk_table_block()."""
    chunks = []
    prose_buffer = []

    def flush_prose():
        if prose_buffer:
            joined = "\n".join(prose_buffer).strip()
            if joined:
                chunks.extend(chunk_text(joined, chunk_size=chunk_size, overlap=overlap))
            prose_buffer.clear()

    for kind, block in split_into_blocks(text):
        if kind == "table":
            flush_prose()
            chunks.extend(chunk_table_block(block, chunk_size))
        else:
            prose_buffer.append(block)
    flush_prose()
    return chunks


def build_rag_prompt(query, context_chunks):
    """Builds the RAG prompt using only the retrieved context."""
    context = "\n\n".join(context_chunks)
    return f"Answer the question using ONLY the information below. If you don't know, say you don't know.\n\nInformation: {context}\n\nQuestion: {query}"


def answer_with_rag(cache, query, top_k=2):
    """Retrieves the top_k most relevant document chunks for query, builds the RAG
    prompt, and sends it to the main LLM (port 8000). Returns (answer, top_docs).
    Also records its internal timing split (embed / retrieval / LLM inference) into
    LAST_TIMINGS - previously the whole function was measured as one block from ask(),
    so retrieval time and LLM inference time could not be separated."""
    t0 = time.time()
    query_vector = get_real_embedding(query)
    t1 = time.time()
    top_docs = cache.retrieve_top_documents(query_vector, top_k=top_k)
    t2 = time.time()
    context_chunks = [content for _, content, _ in top_docs]
    prompt = build_rag_prompt(query, context_chunks)
    answer = get_llm_response(prompt)
    t3 = time.time()
    LAST_TIMINGS["rag_embed"] = t1 - t0
    LAST_TIMINGS["retrieval"] = t2 - t1
    LAST_TIMINGS["llm_inference"] = t3 - t2
    LAST_TIMINGS["chunks_used"] = len(top_docs)
    LAST_TIMINGS["sources"] = sorted({source_file for _, _, source_file in top_docs})
    return answer, top_docs


def ingest_document(cache, content, source_file, chunk_size=None, overlap=None, verbose=True):
    """Splits content into chunks via chunk_text(), embeds each chunk, and stores it.
    Returns the document ids created for this file.

    chunk_size and overlap are in WORDS, but are now chosen from a TOKEN budget measured
    on this document rather than passed in by hand. Before this, every new file needed a
    hand-tuned chunk_size: the default of 500 crashed the embedding server on the first
    real regulation, and 120 was found by trial and error on that one file with no reason
    to expect it to transfer. Passing chunk_size explicitly still overrides everything,
    which is what the controlled experiments use.

    After choosing a size, the longest chunk is checked against the model's real limit
    and the size is reduced if needed. Only the longest chunk is measured (longest by
    characters), because checking every chunk would cost one tokenizer call each for no
    extra safety."""
    if chunk_size is None:
        chunk_size = pick_chunk_size(content)
        for _ in range(3):
            if overlap is None or overlap >= chunk_size:
                overlap = max(chunk_size // 5, 1)
            chunks = chunk_document(content, chunk_size=chunk_size, overlap=overlap)
            if not chunks:
                break
            longest = max(chunks, key=len)
            longest_tokens = count_tokens(longest)
            if longest_tokens <= EMBEDDING_MAX_TOKENS:
                break
            # Scale down by the amount we overshot, plus 10% margin, and re-check.
            chunk_size = max(int(chunk_size * EMBEDDING_MAX_TOKENS / longest_tokens * 0.9), 20)
            overlap = None
        if verbose:
            print(f"[INGEST] {source_file}: chunk_size={chunk_size} words, overlap={overlap}, "
                  f"longest chunk {longest_tokens} tokens (limit {EMBEDDING_MAX_TOKENS})")

    if overlap is None:
        overlap = max(chunk_size // 5, 1)

    chunks = chunk_document(content, chunk_size=chunk_size, overlap=overlap)
    doc_ids = []
    for chunk in chunks:
        embedding = get_real_embedding(chunk)
        doc_id = cache.save_document(chunk, embedding, source_file=source_file)
        doc_ids.append(doc_id)
    return doc_ids



EE_DOCUMENTS_DIR = "ee_documents"


def ingest_folder(cache, folder=EE_DOCUMENTS_DIR, verbose=True):
    """Brings the database in line with the .txt files in folder.

    Each file falls into one of three cases:
      new       - not seen before, so ingest it;
      unchanged - the stored hash matches, so skip it and pay nothing;
      changed   - the hash differs, so drop its old chunks AND every cached answer that
                  was built from it, then ingest the new text.

    The third case is the reason this function tracks hashes at all. Without it, editing
    a document left the old chunks in place and, worse, left cached answers quoting text
    that no longer exists - returned instantly and with no indication that they were
    stale. That is a correctness problem, not a performance one.

    Returns {source_file: number_of_chunks} for the files ingested in this call."""
    if not os.path.isdir(folder):
        print(f"[INGEST] Folder '{folder}' not found - no documents loaded.")
        return {}

    known_hashes = cache.document_hashes()
    ingested = {}

    for filename in sorted(os.listdir(folder)):
        if not filename.lower().endswith(".txt"):
            continue
        path = os.path.join(folder, filename)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        if not content.strip():
            print(f"[INGEST] {filename}: empty file, skipped.")
            continue

        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        stored_hash = known_hashes.get(filename)

        if stored_hash == content_hash:
            if verbose:
                print(f"[INGEST] {filename}: unchanged, skipped.")
            continue

        # A database written before change-detection existed has chunks but no hash
        # record. Without this check those files look new, and their passages would be
        # ingested a second time on top of the existing ones instead of replacing them.
        cache.cursor.execute("SELECT COUNT(*) FROM documents WHERE source_file = ?", (filename,))
        has_old_chunks = cache.cursor.fetchone()[0] > 0

        if stored_hash is not None or has_old_chunks:
            chunks_removed, answers_removed = cache.forget_document(filename)
            reason = "CHANGED since it was loaded" if stored_hash is not None else "already present without a version record"
            print(f"[INGEST] {filename}: {reason} - removed "
                  f"{chunks_removed} old passage(s) and {answers_removed} cached answer(s).")

        doc_ids = ingest_document(cache, content, source_file=filename, verbose=verbose)
        cache.record_document(filename, content_hash)
        ingested[filename] = len(doc_ids)
        if verbose:
            print(f"[INGEST] {filename}: {len(doc_ids)} chunk(s) stored.")

    if verbose:
        cache.cursor.execute("SELECT COUNT(*), COUNT(DISTINCT source_file) FROM documents")
        chunks, sources = cache.cursor.fetchone()
        print(f"[INGEST] Corpus now holds {chunks} chunk(s) from {sources} document(s).")
    return ingested


# =====================================================================================
# Measurement / benchmarking
# =====================================================================================
# Each base question is followed by one or more paraphrases. Order matters: the base
# question is expected to MISS (and be answered via RAG), and its paraphrases are then
# expected to HIT the cache - that is what makes the measured hit rate meaningful
# rather than an artifact of asking the identical string twice.
BENCHMARK_QUERIES = [
    # Each entry carries its ground truth so the same run measures BOTH speed and
    # accuracy. "expect" is a list of GROUPS: the answer counts as correct only if at
    # least one variant from EVERY group appears (case-insensitive). "reject" marks
    # phrases that must NOT appear - used on the trap queries, where quoting the other
    # value proves the wrong cached answer came back.
    #
    # Known limitation: this is string matching, not understanding. The small model
    # sometimes states the correct rule and then draws the opposite conclusion in the
    # same answer (the documented ACME Type C/Type D behaviour); such an answer can
    # still score as correct here. The full answer text is written to the CSV so those
    # cases can be reviewed by hand.

    # --- OSHA: receptacle load, Table S-4 (correct answer: 12 amperes) ---
    {"q": "For a 15-ampere receptacle on a 15- or 20-ampere branch circuit supplying two or more receptacles, what is the maximum cord- and plug-connected load allowed?",
     "expect": [["12 amp", "12 a "]]},
    {"q": "What's the max load on a 15A receptacle when multiple receptacles share the circuit?",
     "expect": [["12 amp", "12 a "]]},

    # --- OSHA: overcurrent protection ratios (3x for a fuse, 6x for a breaker) ---
    {"q": "By what factor may the continuous ampere rating of a fuse exceed the conductor ampacity, and what is the maximum for a breaker's long-time trip setting?",
     "expect": [["three times", "3 times"], ["six times", "6 times"]]},
    {"q": "How many times the conductor ampacity can a fuse rating be?",
     "expect": [["three times", "3 times"]]},

    # --- OSHA: GFCI protection (bathrooms and rooftops) ---
    {"q": "Which receptacles need GFCI protection in bathrooms and on rooftops?",
     "expect": [["bathroom"], ["rooftop", "roof"]]},
    {"q": "Do rooftop outlets require ground-fault protection?",
     # Tightened after reviewing the first graded run: the original label accepted any
     # of yes/require/must/shall, which almost any answer would satisfy. Now the answer
     # has to actually name the protection AND the location.
     "expect": [["ground-fault", "gfci"], ["rooftop", "roof"]]},

    # --- OSHA: clearance above roofs (8.0 ft / 2.44 m) ---
    {"q": "What's the minimum clearance above a roof for overhead conductors?",
     "expect": [["8.0 ft", "8 ft", "2.44 m"]]},

    # --- ACME: motor branch circuit breaker curve (7 kW is above the 5 kW threshold
    #     -> Type D) plus conductor sizing at 125% ---
    {"q": "What type of circuit breaker curve should be used for a 7 kW induction motor branch circuit, and how should the conductors be sized?",
     "expect": [["type d"], ["125%", "125 %", "125 percent"]]},
    {"q": "For a 7 kW motor, is a Type C or Type D breaker correct?",
     "expect": [["type d"]]},

    # --- Off-domain control: the EE system prompt should decline this ---
    {"q": "What's the best programming language for web development?",
     "expect": [["outside", "does not cover", "doesn't cover", "not covered", "cannot", "can't", "sorry"]]},

    # --- Numeric-spec traps. Wording is near-identical to questions already cached
    #     above and only the number differs, but the correct answers differ too, so a
    #     cache HIT here is a FALSE POSITIVE rather than a success. "reject" catches
    #     the cached-but-wrong value being returned.
    # Note: graded on the VALUE only. In one run the model gave the right number but
    # attributed it to Table S-5 instead of Table S-4 - worth knowing about, but the
    # question doesn't ask for a citation, so it isn't failed for that here.
    {"q": "What's the max load on a 30A receptacle when multiple receptacles share the circuit?",
     "expect": [["24 amp", "24 a "]], "reject": ["12 amp"]},
    {"q": "For a 3 kW motor, is a Type C or Type D breaker correct?",
     "expect": [["type c"]], "reject": ["type d curve breaker should", "type d is required", "type d breaker is correct"]},

    # --- OSHA 1910.303: working space depth table (Table S-1) ---
    {"q": "What is the minimum width of working space in front of electric equipment rated 600 V or less?",
     "expect": [["762", "30 in", "30-inch", "30 inch"]]},

    # --- OSHA 1910.303: high-voltage working space table (Table S-2), a specific
    #     cell (9001 V-25 kV, Condition C) rather than the first/most obvious row -
    #     tests whether retrieval finds the right row, not just the right table.
    {"q": "For equipment between 9001 V and 25 kV under Condition C, what is the minimum clear working space?",
     "expect": [["2.8 m", "9.0 ft", "9 ft"]]},

    # --- Not covered by any ingested document. Correct behavior is declining rather
    #     than inventing a number - tests the "say so if it doesn't cover the
    #     question" instruction in DOMAIN_SYSTEM_PROMPT / build_rag_prompt, not just
    #     the off-domain-topic case already covered by query 10.
    # Label widened after a real run: the system correctly declined with "I don't have
    # that information", which the original phrase list missed - a grader false
    # negative, not a system failure.
    {"q": "What is the maximum allowed voltage drop percentage for a branch circuit?",
     "expect": [["outside", "does not cover", "doesn't cover", "not covered", "cannot", "can't",
                 "sorry", "no information", "not specified", "not provided",
                 "don't have", "do not have", "does not specify", "doesn't specify",
                 "not mentioned", "does not mention", "doesn't mention", "unable to"]]},
]

BENCHMARK_CSV = "benchmark_results.csv"


def grade_answer(answer, spec):
    """Checks an answer against its ground truth. Returns (verdict, reason) where
    verdict is "correct", "wrong" or "" (ungraded - no ground truth given).
    Correct means: every "expect" group has at least one variant present, AND no
    "reject" phrase is present."""
    if not spec.get("expect") and not spec.get("reject"):
        return "", ""
    text = answer.lower()
    for phrase in spec.get("reject", []):
        if phrase.lower() in text:
            return "wrong", f"contains rejected phrase '{phrase}'"
    for group in spec.get("expect", []):
        if not any(variant.lower() in text for variant in group):
            return "wrong", f"missing any of {group}"
    return "correct", ""


def run_benchmark(cache, queries=None, csv_path=BENCHMARK_CSV, verbose=False, label="",
                  repeats=1, clear_between=True):
    """Runs each query through ask() and collects BOTH the timings that ask() and
    answer_with_rag() record in LAST_TIMINGS, and an accuracy verdict from
    grade_answer(). Prints a summary and APPENDS every row to a CSV file, so results
    accumulate across runs/days instead of scrolling past in the console.

    repeats > 1 runs the whole set several times in one go. This matters because LLM
    inference time on CPU varies a lot between runs (37x/33x/73x/34x speedups measured
    on identical code), and because the model does not always answer a given question
    the same way - so a single run tells you little about either. Between repeats the
    CACHE is cleared but the documents table is left alone, so every repeat starts cold
    (same miss/hit pattern) without paying to re-embed the corpus.

    verbose=False by default: ask()'s own per-call prints are suppressed so the
    benchmark output stays readable - the numbers are collected, not printed twice."""
    if queries is None:
        queries = BENCHMARK_QUERIES

    run_id = time.strftime("%Y-%m-%d_%H:%M:%S")
    all_rows = []

    print(f"[BENCHMARK] {len(queries)} queries x {repeats} repeat(s) (run_id={run_id}, label={label or 'none'})")
    for repeat in range(1, repeats + 1):
        if repeat > 1 and clear_between:
            cache.clear()
        if repeats > 1:
            print(f"--- repeat {repeat}/{repeats} ---")
        for i, spec in enumerate(queries, start=1):
            query = spec["q"]
            answer, was_hit = ask(cache, query, verbose=verbose)
            t = dict(LAST_TIMINGS)
            verdict, reason = grade_answer(answer, spec)
            row = {
                "run_id": run_id,
                "label": label,
                "repeat": repeat,
                "query_no": i,
                "query": query,
                "cache_hit": was_hit,
                "path": t.get("path", ""),
                "verdict": verdict,
                "grade_reason": reason,
                "similarity": round(t["similarity"], 4) if "similarity" in t else "",
                "total_s": round(t.get("total", 0.0), 3),
                "embed_s": round(t.get("embed", 0.0), 3),
                "cache_search_s": round(t.get("cache_search", 0.0), 4),
                "canonicalize_s": round(t.get("canonicalize", 0.0), 3),
                "retrieval_s": round(t.get("retrieval", 0.0), 4),
                "llm_inference_s": round(t.get("llm_inference", 0.0), 3),
                "chunks_used": t.get("chunks_used", ""),
                "answer": answer.strip(),
            }
            all_rows.append(row)
            status = f"HIT ({row['path']})" if was_hit else "MISS (RAG)"
            mark = {"correct": "OK  ", "wrong": "WRONG", "": "    "}[verdict]
            print(f"  [{i:2}] {status:22} {mark:6} total={row['total_s']:7.3f}s  |  {query[:46]}...")
            if verdict == "wrong":
                print(f"       -> {reason}")

    _write_benchmark_csv(all_rows, csv_path)
    _print_benchmark_summary(all_rows, csv_path)
    if repeats > 1:
        _print_consistency_report(all_rows, queries, repeats)
    return all_rows


def report_chunk_rank(cache, query, needle, max_rank=30, restrict_to_top_source=True):
    """Deterministic recall diagnostic: finds where the chunk actually CONTAINING the
    answer ranks for this query. No LLM call, so it isolates retrieval from generation.

    A wrong answer can come from two very different places - the right chunk was
    retrieved and the model misread it, or the right chunk never made it into the
    context. top_k experiments confound the two; this separates them. If the answer
    chunk ranks below the current top_k, it is a recall problem and raising top_k can
    fix it. If it ranks inside top_k and the answer is still wrong, raising top_k will
    not help and the problem is in the prompt or the model."""
    query_vector = get_real_embedding(query)
    ranked = cache.retrieve_top_documents(
        query_vector, top_k=max_rank, restrict_to_top_source=restrict_to_top_source
    )
    print(f"\n[CHUNK RANK] Looking for the chunk containing {needle!r}")
    print(f"  Query: {query[:66]}...")
    for rank, (score, content, source_file) in enumerate(ranked, start=1):
        if needle.lower() in content.lower():
            verdict = "INSIDE default top_k=4" if rank <= 4 else "OUTSIDE default top_k=4 - recall problem"
            print(f"  Found at rank {rank}/{len(ranked)} (score={score:.4f}, source={source_file}) -> {verdict}")
            return rank
    print(f"  NOT FOUND in the top {len(ranked)} chunks of the selected source.")
    print("  -> Either the answer text is split across a chunk boundary, or source")
    print("     scoping picked the wrong document for this query.")
    return None


OVERRIDE_LINE = (
    "The information provided below is authoritative for this task. If it disagrees "
    "with what you already know about electrical practice, follow the provided "
    "information and not your own knowledge."
)


def run_prompt_experiment(cache, spec, trials=5, top_k=8):
    """Tests whether an explicit instruction can stop the model overriding the
    retrieved text with its own prior knowledge.

    Measured on 10 September: the ACME manual was edited from 125% to 130%, the change
    reached both stored chunks, and the model still answered 125% - the real NEC value
    it knows from training. A control question about the manual's arbitrary section
    number was answered correctly, so retrieval is fine and the conflict is at
    generation time.

    Three variants, because where an instruction sits usually matters more than its
    wording: in the system message, next to the context, or both. Retrieval is
    identical in all three and the cache is bypassed, so the only thing that varies is
    the instruction."""
    query_vector = get_real_embedding(spec["q"])
    top_docs = cache.retrieve_top_documents(query_vector, top_k=top_k)
    context_chunks = [content for _, content, _ in top_docs]
    base_prompt = build_rag_prompt(spec["q"], context_chunks)

    variants = [
        ("current prompt (no override)", DOMAIN_SYSTEM_PROMPT, base_prompt),
        ("override in system message", DOMAIN_SYSTEM_PROMPT + " " + OVERRIDE_LINE, base_prompt),
        ("override next to the context", DOMAIN_SYSTEM_PROMPT, OVERRIDE_LINE + "\n\n" + base_prompt),
        ("override in both places", DOMAIN_SYSTEM_PROMPT + " " + OVERRIDE_LINE, OVERRIDE_LINE + "\n\n" + base_prompt),
    ]

    print(f"\n[PROMPT EXPERIMENT] {trials} trials per variant, {len(top_docs)} chunk(s) retrieved")
    print(f"  Question: {spec['q'][:70]}...")
    results = {}
    for name, system_prompt, user_prompt in variants:
        correct = 0
        for _ in range(trials):
            answer = get_llm_response(user_prompt, system_prompt=system_prompt)
            verdict, _ = grade_answer(answer, spec)
            if verdict == "correct":
                correct += 1
        results[name] = correct
        print(f"  {name:32} {correct}/{trials} correct")

    best, worst = max(results.values()), min(results.values())
    if best == 0:
        print("  -> No variant helped. The model keeps its own answer whatever it is told,")
        print("     which makes this a model limitation rather than a prompt problem.")
    elif worst == trials:
        print("  -> Every variant was correct, including the current prompt, so this run")
        print("     says nothing about the instruction - the question was not failing here.")
    elif best - worst >= trials - 1:
        print("  -> A clear winner: the instruction changes the outcome.")
    else:
        print("  -> Mixed. The instruction helps but does not settle it; more trials needed")
        print("     before changing the prompt on this evidence.")
    return results


def run_ab_experiment(cache, spec, variants, trials=5, answer_rank=None):
    """Controlled A/B experiment: asks the SAME question repeatedly under different
    retrieval settings and reports how often each setting produces a correct answer.

    Needed because a 3-repeat benchmark cannot separate "this setting causes the
    error" from "the model is simply inconsistent here" - the ACME 7 kW question
    scored 3/3 correct under pooled retrieval and then 0/3 under source-scoped
    retrieval, which looks causal but is well within what sampling noise can produce
    at temperature 0.3. This calls answer_with_rag() directly (bypassing the cache) so
    that every trial actually re-runs the LLM instead of replaying a cached answer.

    variants: list of (name, kwargs) passed through to retrieve_top_documents."""
    print(f"\n[A/B EXPERIMENT] {trials} trials per variant")
    print(f"  Question: {spec['q'][:70]}...")
    results = {}
    for name, kwargs in variants:
        correct = 0
        sources_seen = set()
        for _ in range(trials):
            query_vector = get_real_embedding(spec["q"])
            top_docs = cache.retrieve_top_documents(query_vector, **kwargs)
            sources_seen.update(sf for _, _, sf in top_docs)
            prompt = build_rag_prompt(spec["q"], [c for _, c, _ in top_docs])
            answer = get_llm_response(prompt)
            verdict, _ = grade_answer(answer, spec)
            if verdict == "correct":
                correct += 1
        results[name] = correct
        print(f"  {name:34} {correct}/{trials} correct   chunks={len(top_docs)} sources={sources_seen}")

    values = list(results.values())
    if max(values) - min(values) >= trials - 1:
        print("  -> Large, consistent gap between variants: the setting looks causal.")
    elif max(values) == min(values):
        # Guard against a misreading that already happened once: a top_k sweep where
        # EVERY variant scored 0 was reported as "the setting makes no difference",
        # when in fact the chunk holding the answer ranked 13th and no variant went
        # that far - so the hypothesis was never actually tested. If a rank is known
        # and sits beyond every top_k tried, say so instead of claiming a null result.
        tried_top_k = [kw.get("top_k") for _, kw in variants if kw.get("top_k")]
        if answer_rank is not None and tried_top_k and answer_rank > max(tried_top_k):
            print(f"  -> INCONCLUSIVE: the answer chunk ranks {answer_rank}, beyond every")
            print(f"     top_k tried (max {max(tried_top_k)}), so no variant could include it.")
            print(f"     Re-run with top_k >= {answer_rank} to actually test this.")
        else:
            print("  -> Identical scores: the setting makes no difference here.")
    else:
        print("  -> Mixed scores in every variant: the model is inconsistent on this")
        print("     question regardless of the setting, so earlier 3/3 and 0/3 runs were")
        print("     most likely sampling noise rather than evidence about retrieval.")
    return results


def _print_consistency_report(rows, queries, repeats):
    """Per-query accuracy across repeats. This is the point of repeating: a question
    the model answers correctly 3 times out of 5 is a very different finding from one
    it always gets right or always gets wrong, and a single run cannot tell them
    apart."""
    print("\n[CONSISTENCY ACROSS REPEATS]")
    flaky = []
    for i in range(1, len(queries) + 1):
        graded = [r for r in rows if r["query_no"] == i and r["verdict"]]
        if not graded:
            continue
        correct = sum(1 for r in graded if r["verdict"] == "correct")
        flag = ""
        if 0 < correct < len(graded):
            flag = "  <-- INCONSISTENT"
            flaky.append(i)
        elif correct == 0:
            flag = "  <-- always wrong"
        print(f"  Q{i:<2} {correct}/{len(graded)} correct{flag}   {queries[i - 1]['q'][:44]}...")
    if flaky:
        print(f"  Inconsistent queries: {flaky} - same question, same documents, different verdicts")


def _write_benchmark_csv(rows, csv_path):
    """Appends rows to csv_path, writing the header only when creating the file."""
    is_new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if is_new_file:
            writer.writeheader()
        writer.writerows(rows)


def _print_benchmark_summary(rows, csv_path):
    """Prints the aggregate metrics: cache hit rate plus mean timings, split by
    hit/miss (mixing them would be misleading - a hit skips the LLM entirely)."""
    def mean(values):
        values = [v for v in values if v]
        return sum(values) / len(values) if values else 0.0

    hits = [r for r in rows if r["cache_hit"]]
    misses = [r for r in rows if not r["cache_hit"]]

    print("\n[BENCHMARK SUMMARY]")
    print(f"  Queries              : {len(rows)}")
    print(f"  Cache hit rate       : {len(hits) / len(rows) * 100:.1f}%  ({len(hits)} hit / {len(misses)} miss)")
    if hits:
        print(f"  Mean response (hit)  : {mean([r['total_s'] for r in hits]):.3f}s")
        print(f"    - embedding        : {mean([r['embed_s'] for r in hits]):.3f}s")
        print(f"    - semantic search  : {mean([r['cache_search_s'] for r in hits]):.4f}s")
    if misses:
        print(f"  Mean response (miss) : {mean([r['total_s'] for r in misses]):.3f}s")
        print(f"    - canonicalize     : {mean([r['canonicalize_s'] for r in misses]):.3f}s")
        print(f"    - RAG retrieval    : {mean([r['retrieval_s'] for r in misses]):.4f}s")
        print(f"    - LLM inference    : {mean([r['llm_inference_s'] for r in misses]):.3f}s")
    if hits and misses:
        speedup = mean([r["total_s"] for r in misses]) / max(mean([r["total_s"] for r in hits]), 1e-9)
        repeats_seen = len({r.get("repeat", 1) for r in rows})
        note = f"averaged over {repeats_seen} repeats" if repeats_seen > 1 else "SINGLE run - LLM time varies a lot, average several runs"
        print(f"  Hit vs miss speedup  : {speedup:.0f}x  ({note})")

    graded = [r for r in rows if r["verdict"]]
    if graded:
        correct = [r for r in graded if r["verdict"] == "correct"]
        print(f"  Accuracy             : {len(correct) / len(graded) * 100:.1f}%  ({len(correct)}/{len(graded)} graded)")
        # A hit that returns a wrong answer is worse than a miss - it is served
        # instantly and confidently, so it is worth reporting separately rather than
        # letting the hit rate above imply it was a success.
        bad_hits = [r for r in graded if r["cache_hit"] and r["verdict"] == "wrong"]
        print(f"    - wrong cache hits : {len(bad_hits)}")
        for r in [r for r in graded if r["verdict"] == "wrong"]:
            print(f"    - Q{r['query_no']} wrong ({'hit' if r['cache_hit'] else 'miss'}): {r['grade_reason']}")
    print(f"  Rows appended to     : {csv_path}")


# =====================================================================================
# Proactive / off-peak cache pre-warming (under an infinite-memory assumption)
# =====================================================================================
# The reactive cache (above) only caches questions that are ACTUALLY asked. This
# section adds a separate layer that decides, ahead of time, which question-entity
# combinations should be cached during off-peak hours. Four sub-steps:
#   1) template/slot definition (this block)
#   2) candidate combination list
#   3) prioritization
#   4) off-peak pre-warm function

# --- Template/slot definition -------------------------------------------------------
# Design decision: instead of parsing free text backward into a template (reverse-
# parsing), we generate forward (template -> concrete question). Two reasons:
#   1) We don't have real traffic logs yet (local/demo system) - there's no real pool
#      of questions to reverse-parse.
#   2) The concrete questions generated this way work directly with the search()/
#      entities_agree() machinery already in place (see the "already cached?" check
#      in step 4 below) - no separate text-matching/parsing layer is needed.
# As real user logs accumulate, it will be easy to add a reverse-parse layer ("match
# the incoming question to the closest template") while keeping the same template
# dictionary - out of scope for now.
#
# Slot: only "country" is modeled for now - consistent with the existing
# ENTITY_ALIASES and test set. The same pattern can later be extended to other slots,
# e.g. exam type (IELTS/TOEFL) or degree level (bachelor's/master's).

QUESTION_TEMPLATES = [
    {"id": "ielts_min_score", "template": "What is the minimum IELTS score required for a master's degree in {country}?"},
    {"id": "toefl_min_score", "template": "What is the minimum TOEFL score required for a master's degree in {country}?"},
    {"id": "visa_documents", "template": "What documents do I need for a student visa application to {country}?"},
    {"id": "tuition_fee", "template": "What is the average tuition fee for international students in {country}?"},
    {"id": "gpa_requirement", "template": "What is the minimum GPA required for a master's degree application in {country}?"},
    {"id": "application_deadline", "template": "What is the typical application deadline for master's programs in {country}?"},
    {"id": "scholarships", "template": "What scholarship opportunities are available for international students in {country}?"},
    {"id": "health_insurance", "template": "Is health insurance mandatory for international students in {country}?"},
    {"id": "visa_processing_time", "template": "What is the typical visa processing time for {country}?"},
    {"id": "part_time_work", "template": "What are the part-time work rules for international students in {country}?"},
    # Week-2 (Task 3) expansion - 20 new templates. All passed the 435-pair
    # topic_keywords overlap check (see the TOPIC_KEYWORDS note).
    {"id": "living_cost", "template": "What is the average monthly living cost for international students in {country}?"},
    {"id": "student_visa_financial_proof", "template": "How much financial proof is required for a student visa application to {country}?"},
    {"id": "visa_interview", "template": "Is a visa interview required for international students applying to {country}?"},
    {"id": "acceptance_rate", "template": "What is the typical acceptance rate for universities in {country}?"},
    {"id": "language_waiver", "template": "Can international students obtain an IELTS or TOEFL waiver in {country}?"},
    {"id": "application_fee", "template": "What is the average university application fee in {country}?"},
    {"id": "student_accommodation", "template": "What accommodation options are available for international students in {country}?"},
    {"id": "post_study_work_visa", "template": "Can international students obtain a post-study work visa after graduation in {country}?"},
    {"id": "permanent_residency_pathway", "template": "Does studying in {country} provide a pathway to permanent residency?"},
    {"id": "bank_account", "template": "Can international students open a bank account in {country}?"},
    {"id": "student_discount", "template": "What student discounts and benefits are available in {country}?"},
    {"id": "internship_opportunities", "template": "Are internship opportunities available for international students in {country}?"},
    {"id": "master_duration", "template": "What is the typical duration of a master's degree program in {country}?"},
    {"id": "bachelor_duration", "template": "What is the typical duration of a bachelor's degree program in {country}?"},
    {"id": "credit_transfer", "template": "Can international students transfer academic credits to universities in {country}?"},
    {"id": "intake_periods", "template": "What are the main university intake periods in {country}?"},
    {"id": "conditional_admission", "template": "Do universities in {country} offer conditional admission to international students?"},
    {"id": "campus_work", "template": "Can international students work on campus while studying in {country}?"},
    {"id": "dependents", "template": "Can international students bring dependents while studying in {country}?"},
    {"id": "graduation_employment_rate", "template": "What are the employment prospects for international graduates in {country}?"},
]

COUNTRY_SLOT_VALUES = [
    "the UK", "the US", "Canada", "Germany", "Australia", "New Zealand",
    "the Netherlands", "Sweden", "France", "Ireland", "Italy", "Spain",
    "Switzerland", "Japan", "South Korea", "China",
    "Turkey", "Poland", "Belgium", "Austria",
]


def instantiate_template(template, country):
    """Turns a template into a concrete question for one country value."""
    return template["template"].format(country=country)


# --- Candidate combination list -------------------------------------------------------
def generate_candidate_combinations():
    """Combines every template with every country (cartesian product) to produce the
    full list of concrete pre-warm candidate questions. No ordering/priority yet - that's
    handled next; here we only answer 'which combinations are possible'."""
    candidates = []
    for template in QUESTION_TEMPLATES:
        for country in COUNTRY_SLOT_VALUES:
            candidates.append({
                "question": instantiate_template(template, country),
                "template_id": template["id"],
                "country": country,
            })
    return candidates


# --- Prioritization -----------------------------------------------------------
# Since there's no real traffic log (local/demo system), we use a simple, explainable
# score: template importance x country popularity. The weights are kept in SEPARATE
# dictionaries, untouched by the template/candidate data structures above - so as real
# query logs accumulate, these two dictionaries can easily be swapped for frequency-
# based values without changing the template/candidate generation code at all.
#
# Note (infinite-memory assumption): since this system will never see real traffic,
# the most concrete way to approximate "infinite memory" is to pre-fill as broad a
# combination set as possible during the off-peak window. That's why the template/
# country lists were deliberately kept large. Prioritization still matters: even
# though MEMORY is unlimited, the TIME available in the off-peak window is not (each
# LLM call takes a few seconds) - so processing the "most valuable" combinations
# first still matters.

TEMPLATE_PRIORITY_WEIGHTS = {
    "ielts_min_score": 1.0,        # very frequently asked, a core requirement question
    "visa_documents": 1.0,         # very frequently asked, directly actionable
    "toefl_min_score": 0.9,        # as common as IELTS, an alternative exam
    "tuition_fee": 0.8,            # frequently asked, financial planning
    "gpa_requirement": 0.8,        # frequently asked, a core requirement question
    "application_deadline": 0.75,  # frequently asked, critical for planning
    "scholarships": 0.7,           # asked, but more niche/eligibility-dependent
    "health_insurance": 0.65,      # useful, important since mandatory in some countries
    "visa_processing_time": 0.6,   # useful but more of a logistics detail
    "part_time_work": 0.6,         # useful but a more niche question
    # Week-2 (Task 3) expansion
    "living_cost": 0.85,                    # very frequently asked, central to financial planning
    "student_visa_financial_proof": 0.85,   # a critical/mandatory visa requirement
    "post_study_work_visa": 0.7,            # a very important factor in the decision
    "application_fee": 0.7,
    "acceptance_rate": 0.65,
    "master_duration": 0.65,
    "student_accommodation": 0.6,
    "language_waiver": 0.6,
    "permanent_residency_pathway": 0.6,
    "graduation_employment_rate": 0.6,
    "bachelor_duration": 0.6,
    "visa_interview": 0.55,
    "conditional_admission": 0.55,
    "credit_transfer": 0.5,
    "internship_opportunities": 0.5,
    "campus_work": 0.5,
    "intake_periods": 0.5,
    "bank_account": 0.4,
    "student_discount": 0.35,
    "dependents": 0.35,
}

COUNTRY_PRIORITY_WEIGHTS = {
    "the UK": 1.0,
    "the US": 1.0,
    "Canada": 0.9,
    "Germany": 0.8,
    "Australia": 0.8,
    "New Zealand": 0.65,
    "the Netherlands": 0.6,
    "Sweden": 0.55,
    "France": 0.5,
    "Ireland": 0.5,
    "Italy": 0.45,
    "Spain": 0.45,
    "Switzerland": 0.45,
    "Japan": 0.4,
    "South Korea": 0.35,
    "China": 0.5,  # the country where the project is run locally
    "Turkey": 0.55,
    "Poland": 0.45,
    "Austria": 0.45,
    "Belgium": 0.4,
}


def prioritize_candidates(candidates):
    """Assigns each candidate a priority score as template_weight x country_weight, and
    returns the list sorted highest-to-lowest (returns a new list, doesn't mutate the
    original). Uses a default weight of 0.5 for a template_id/country missing from the
    dictionary - so the lists can be extended without silently breaking if a weight is
    forgotten."""
    scored = []
    for c in candidates:
        template_weight = TEMPLATE_PRIORITY_WEIGHTS.get(c["template_id"], 0.5)
        country_weight = COUNTRY_PRIORITY_WEIGHTS.get(c["country"], 0.5)
        scored.append({**c, "priority": template_weight * country_weight})
    return sorted(scored, key=lambda c: c["priority"], reverse=True)


# --- Off-peak pre-warm function ----------------------------------------------
def prewarm_cache(cache, candidates=None, limit=None, verbose=True):
    """Walks the candidate questions in priority order; for each one, calls
    cache.search() in EXACTLY the same way as the reactive path does - i.e. the same
    'is this already cached?' rules apply, including the raw_query-based entity guard.
    Only goes to the LLM (and writes to the cache) for combinations that are genuinely
    missing. Completely independent of the live query flow; meant to be triggered by
    an off-peak scheduled job (cron / systemd timer / Windows Task Scheduler) - see the
    `--prewarm` flag in __main__.

    If candidates isn't given, uses the full output of steps 1-3 (prioritize_candidates
    + generate_candidate_combinations).
    If limit is given, adds at most that many NEW records - can be left as limit=None
    under the infinite-memory assumption, but useful in a demo/test environment to
    avoid sending the LLM an excessive number of requests."""
    if candidates is None:
        candidates = prioritize_candidates(generate_candidate_combinations())

    stats = {"attempted": 0, "already_cached": 0, "newly_cached": 0}
    for candidate in candidates:
        if limit is not None and stats["newly_cached"] >= limit:
            break
        stats["attempted"] += 1
        question = candidate["question"]

        canonical_query = canonicalize_query(question)
        query_vector = get_real_embedding(canonical_query)

        # Same search() call as the reactive path: is this candidate already covered,
        # semantically, by an earlier real user question or an earlier pre-warm run?
        existing = cache.search(query_vector, raw_query=question, threshold=0.55)
        if existing is not None:
            stats["already_cached"] += 1
            if verbose:
                print(f"[PREWARM SKIP] already in cache (priority={candidate['priority']:.2f}): {question}")
            continue

        answer = get_llm_response(question)
        cache.save(canonical_query, answer, query_vector, raw_prompt=question)
        stats["newly_cached"] += 1
        if verbose:
            print(
                f"[PREWARM ADD]  (priority={candidate['priority']:.2f}, "
                f"template={candidate['template_id']}, country={candidate['country']}): {question}"
            )

    return stats


# =====================================================================================
# "Chatbot" interface: ask a single question, get the cached answer back INSTANTLY
# (even for a paraphrase) WITHOUT going to the LLM at all, if it's already cached.
# =====================================================================================
# This is NOT a new mechanism - it's a thin, reusable wrapper around the reactive flow
# (canonicalize -> embed -> search -> hit/miss). Thanks to the large cache filled by
# pre-warming, most realistically-asked questions can now be answered straight from a
# precomputed answer, without ever touching the LLM.

def ask(cache, question, verbose=True, top_k=8):
    """Answers a single question from the cache. Performs a two-tier lookup:
      1) FAST PATH: without any LLM call, embeds the RAW question directly and checks
         the cache (only hits the embedding server - never the LLM). This alone is
         enough for most repeat/paraphrase cases.
      2) If the fast path misses: canonicalize_query() (an LLM call) standardizes the
         query, re-embeds it, and searches again - some paraphrases end up closer in
         embedding space after canonicalization.
    If neither finds a match, generates a new answer from the LLM and caches it in its
    canonical form (so it can be found via either the fast or the canonical path in the
    future). Returns (answer, was_cache_hit).

    Why this change: in the first version, canonicalize_query() ran on EVERY call (even
    on a HIT) - since that's an LLM call, a "cache hit" took almost as long as a "cache
    miss" in practice (the only real saving was skipping the 2nd LLM call, i.e.
    generating the answer). The fast path lets genuine/practical repeats get answered
    WITHOUT ever hitting the LLM. Cost: on a genuine MISS, one extra embedding call
    (~0.5-1s) - a small price compared to the gain on HITs."""
    t0 = time.time()
    LAST_TIMINGS.clear()

    # 1) Fast path - without the LLM
    raw_vector = get_real_embedding(question)
    t1 = time.time()
    match = cache.search(raw_vector, raw_query=question, threshold=0.55)
    t2 = time.time()
    if match is not None:
        _, answer, similarity = match
        LAST_TIMINGS.update({
            "path": "fast_hit", "cache_hit": True, "similarity": similarity,
            "embed": t1 - t0, "cache_search": t2 - t1, "total": t2 - t0,
        })
        if verbose:
            print(f"[CACHE HIT - fast path] (similarity %{similarity * 100:.1f}) - LLM was never called.")
            print(f"[TIMING] embed={t1 - t0:.2f}s | search={t2 - t1:.2f}s | TOTAL={t2 - t0:.2f}s")
        return answer, True

    # 2) Fast path missed - canonicalize and try again
    canonical_query = canonicalize_query(question)
    t3 = time.time()
    canonical_vector = get_real_embedding(canonical_query)
    t4 = time.time()
    match = cache.search(canonical_vector, raw_query=question, threshold=0.55)
    t5 = time.time()
    if match is not None:
        _, answer, similarity = match
        LAST_TIMINGS.update({
            "path": "canonical_hit", "cache_hit": True, "similarity": similarity,
            "embed": (t1 - t0) + (t4 - t3), "canonicalize": t3 - t2,
            "cache_search": (t2 - t1) + (t5 - t4), "total": t5 - t0,
        })
        if verbose:
            print(f"[CACHE HIT - canonical path] (similarity %{similarity * 100:.1f})")
            print(f"[TIMING] fast_path_miss={t2 - t0:.2f}s | canonicalize={t3 - t2:.2f}s | embed={t4 - t3:.2f}s | search={t5 - t4:.2f}s | TOTAL={t5 - t0:.2f}s")
        return answer, True

    # 3) Genuinely not cached - ask the LLM via RAG and cache it in canonical form
    # top_k default raised 4 -> 8 on measured evidence: the working-space-width query
    # failed at top_k=4 because the chunk holding its answer ranked 5th, one place
    # outside the window, and scored 3/3 correct at top_k=8. Not raised further because
    # more context is not monotonically better (top_k=12 scored 2/3 on the same query)
    # and every extra chunk costs LLM inference time, which already dominates a miss.
    answer, top_docs = answer_with_rag(cache, question, top_k=top_k)
    t6 = time.time()
    cache.save(canonical_query, answer, canonical_vector, raw_prompt=question,
               source_files=LAST_TIMINGS.get("sources"))
    LAST_TIMINGS.update({
        "path": "miss", "cache_hit": False,
        "embed": (t1 - t0) + (t4 - t3), "canonicalize": t3 - t2,
        "cache_search": (t2 - t1) + (t5 - t4), "rag_total": t6 - t5, "total": t6 - t0,
    })
    if verbose:
        print("[CACHE MISS] - asked the LLM via RAG, generated an answer, and cached it.")
        if top_docs:
            print(f"[CACHE MISS] RAG used {len(top_docs)} chunk(s), top source: {top_docs[0][2]}")
        print(f"[TIMING] fast_path_miss={t2 - t0:.2f}s | canonical_path_miss={t5 - t2:.2f}s | llm_answer={t6 - t5:.2f}s | TOTAL={t6 - t0:.2f}s")
    return answer, False


def _batch_cosine_similarity(query_vector, stored_vectors_matrix):
    """Mathematical/vector-based similarity computation. The previous implementation
    looped over EVERY row in the cache one at a time, calling np.dot separately for
    each (N separate operations). This function instead stacks all stored vectors into
    a single (N x D) matrix once, then computes all N similarity scores at once with a
    SINGLE matrix-vector multiplication ((N x D) @ (D,) = (N,)).
    Both the query vector and every stored row are explicitly L2-normalized before the
    multiplication, so the result is a TRUE cosine similarity - sim(q,v) = (q.v) /
    (||q|| * ||v||) - rather than a raw dot product. This does not depend on the
    embedding server already returning unit-norm vectors (an implicit, unverified
    assumption in the earlier version); normalizing here guarantees it regardless.
    A small epsilon avoids division by zero for a theoretical all-zero vector.
    Verified by regression testing: normalization does not change any HIT/MISS
    decision on the existing validated test cases (see project notes), and the old
    (loop) vs new (matrix) methods give the exact same index/order for both the
    single best match (search/best_match_score) and ranked top-k (RAG retrieval)."""
    query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-8)
    stored_norms = np.linalg.norm(stored_vectors_matrix, axis=1, keepdims=True)
    stored_normalized = stored_vectors_matrix / (stored_norms + 1e-8)
    return stored_normalized @ query_norm


class LocalSemanticCache:
    def __init__(self, db_path="semantic_cache.db", max_records=MAX_CACHE_RECORDS):
        self.db_path = db_path
        self.max_records = max_records
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.create_table()
        self._migrate_last_accessed_column()
        self._migrate_raw_prompt_column()
        self.create_documents_table()
        # In-memory copy of (ids, questions, answers, raw_prompts, embedding matrix)
        # for the cache table. None means "stale, rebuild on next read". Benchmarking
        # showed that re-fetching from SQLite and re-stacking the embeddings into a
        # matrix on EVERY search() call is 80-90% of that call's total time - far more
        # than the matrix multiplication itself (see project notes). Since a real
        # session is read-heavy (many searches per write, especially in --chat mode),
        # caching this in memory and only invalidating it on a write (save/evict/clear)
        # removes that repeated cost for the common case.
        self._matrix_cache = None

    def create_table(self):
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT,
                answer TEXT,
                embedding BLOB,
                last_accessed REAL,
                raw_prompt TEXT,
                source_files TEXT
            )
        """)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents_meta (
                source_file TEXT PRIMARY KEY,
                content_hash TEXT,
                ingested_at REAL
            )
        """)
        # A database created before source_files existed still has the old cache table,
        # and CREATE TABLE IF NOT EXISTS will not add the column. Add it here so an
        # existing cache keeps working instead of failing on every save.
        self.cursor.execute("PRAGMA table_info(cache)")
        if "source_files" not in {row[1] for row in self.cursor.fetchall()}:
            self.cursor.execute("ALTER TABLE cache ADD COLUMN source_files TEXT")
        self.conn.commit()

    def _migrate_raw_prompt_column(self):
        # The entity guard now runs on the user's original question rather than
        # canonicalize_query()'s (a small, lossy local model) output - see search().
        # Older semantic_cache.db files (from earlier weeks) may not have this column.
        self.cursor.execute("PRAGMA table_info(cache)")
        columns = [row[1] for row in self.cursor.fetchall()]
        if "raw_prompt" not in columns:
            self.cursor.execute("ALTER TABLE cache ADD COLUMN raw_prompt TEXT")
            self.conn.commit()

    def _migrate_last_accessed_column(self):
        # Older semantic_cache.db files (pre-LRU) won't have this column yet.
        self.cursor.execute("PRAGMA table_info(cache)")
        columns = [row[1] for row in self.cursor.fetchall()]
        if "last_accessed" not in columns:
            self.cursor.execute("ALTER TABLE cache ADD COLUMN last_accessed REAL")
            self.cursor.execute(
                "UPDATE cache SET last_accessed = ? WHERE last_accessed IS NULL",
                (time.time(),)
            )
            self.conn.commit()

    def save(self, question, answer, embedding, raw_prompt=None, source_files=None):
        """source_files records which documents the answer was built from, so that when
        one of those documents changes the answers derived from it can be dropped
        instead of being served from a version of the text that no longer exists."""
        joined = ",".join(sorted(source_files)) if source_files else None
        self.cursor.execute(
            "INSERT INTO cache (question, answer, embedding, last_accessed, raw_prompt, source_files) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (question, answer, embedding.tobytes(), time.time(), raw_prompt, joined)
        )
        self.conn.commit()
        self._matrix_cache = None  # invalidate: next read rebuilds it
        self._evict_if_needed()

    def _get_matrix_cache(self):
        """Returns (ids, questions, answers, raw_prompts, matrix) for the whole cache
        table, rebuilding from SQLite only if the in-memory copy is missing (i.e. a
        write happened since the last build - see save()/_evict_if_needed()/clear()).
        matrix is None if the cache is empty."""
        if self._matrix_cache is None:
            self.cursor.execute("SELECT id, question, answer, embedding, raw_prompt FROM cache")
            rows = self.cursor.fetchall()
            if not rows:
                self._matrix_cache = ([], [], [], [], None)
            else:
                ids = [r[0] for r in rows]
                questions = [r[1] for r in rows]
                answers = [r[2] for r in rows]
                raw_prompts = [r[4] for r in rows]
                matrix = np.stack([np.frombuffer(r[3], dtype=np.float32) for r in rows])
                self._matrix_cache = (ids, questions, answers, raw_prompts, matrix)
        return self._matrix_cache

    def search(self, query_vector, raw_query=None, threshold=0.55, min_entity_overlap=0.5):
        """The entity guard runs on the user's ORIGINAL question (raw_query) and that
        cache entry's original question (the raw_prompt column), instead of
        canonicalize_query()'s output. canonicalize_query() is a small/lossy local
        model, so it could drop or alter a country/exam name and trip the guard by
        mistake (see the domain tests' 6/6-MISS finding). For older (pre-raw_prompt)
        records, falls back to the canonical question text if that column is empty.

        Best match is found via a SINGLE matrix multiplication (_batch_cosine_similarity)
        rather than N separate np.dot calls - see that function's docstring. The matrix
        itself comes from _get_matrix_cache(), which avoids re-reading SQLite on every
        call - see its docstring."""
        ids, questions, answers, raw_prompts, stored_matrix = self._get_matrix_cache()
        if stored_matrix is None:
            return None

        similarities = _batch_cosine_similarity(query_vector, stored_matrix)
        best_idx = int(np.argmax(similarities))
        max_similarity = float(similarities[best_idx])
        best_match = (questions[best_idx], answers[best_idx], max_similarity)
        best_id = ids[best_idx]
        best_raw_prompt = raw_prompts[best_idx]

        if max_similarity >= threshold:
            if raw_query is not None:
                cached_text_for_entities = best_raw_prompt if best_raw_prompt else best_match[0]
                entity_ok = entities_agree(raw_query, cached_text_for_entities, min_entity_overlap)
                topic_ok = topics_agree(raw_query, cached_text_for_entities, min_entity_overlap)
                specs_ok = specs_agree(raw_query, cached_text_for_entities)
                agree = entity_ok and topic_ok and specs_ok
                if VERBOSE_DIAGNOSTICS:
                    # Show EXACTLY what the guard is comparing - so that on an
                    # unexpected HIT/MISS we can see the root cause instead of
                    # guessing at it.
                    print(
                        f"[DEBUG-ENTITY] sim={max_similarity * 100:.2f}% | raw_query='{raw_query}' | cached='{cached_text_for_entities}' "
                        f"(raw_prompt {'found' if best_raw_prompt else 'MISSING - fell back to canonical text'}) "
                        f"| q_ent={extract_entities(raw_query)} | c_ent={extract_entities(cached_text_for_entities)} | entity_ok={entity_ok} "
                        f"| q_topic={extract_topic_keywords(raw_query)} | c_topic={extract_topic_keywords(cached_text_for_entities)} | topic_ok={topic_ok} "
                        f"| q_spec={extract_numeric_specs(raw_query)} | c_spec={extract_numeric_specs(cached_text_for_entities)} | specs_ok={specs_ok} "
                        f"| agree={agree}"
                    )
                if not agree:
                    return None  # embeddings agreed, but named entities didn't - treat as a miss
            self._touch(best_id)
            return best_match
        return None

    def best_match_score(self, query_vector):
        """Calibration helper: returns the nearest neighbor (question, similarity)
        WITHOUT applying the threshold; returns None if the cache is empty. search()
        doesn't expose the score on a MISS, so this is used to see the similarity gap
        between genuine paraphrases and entity-swap traps in a new domain (and to
        recalibrate threshold=0.55 if needed).

        Uses the same in-memory matrix cache as search() - see _get_matrix_cache()."""
        ids, questions, answers, raw_prompts, stored_matrix = self._get_matrix_cache()
        if stored_matrix is None:
            return None
        similarities = _batch_cosine_similarity(query_vector, stored_matrix)
        best_idx = int(np.argmax(similarities))
        return questions[best_idx], float(similarities[best_idx])

    def _touch(self, row_id):
        """Refresh last_accessed on a cache hit, so LRU eviction spares hot entries."""
        self.cursor.execute("UPDATE cache SET last_accessed = ? WHERE id = ?", (time.time(), row_id))
        self.conn.commit()

    def _evict_if_needed(self):
        self.cursor.execute("SELECT COUNT(*) FROM cache")
        count = self.cursor.fetchone()[0]
        if count > self.max_records:
            overflow = count - self.max_records
            self.cursor.execute(
                "DELETE FROM cache WHERE id IN (SELECT id FROM cache ORDER BY last_accessed ASC LIMIT ?)",
                (overflow,)
            )
            self.conn.commit()
            self._matrix_cache = None  # invalidate: rows were removed

    def stats(self):
        self.cursor.execute("SELECT COUNT(*) FROM cache")
        return {"records": self.cursor.fetchone()[0], "max_records": self.max_records}

    def clear(self):
        """Wipe all cached records without deleting the underlying db file."""
        self.cursor.execute("DELETE FROM cache")
        try:
            self.cursor.execute("DELETE FROM sqlite_sequence WHERE name = 'cache'")
        except sqlite3.OperationalError:
            pass  # sqlite_sequence doesn't exist yet if nothing was ever inserted
        self.conn.commit()
        self._matrix_cache = None

    # --- Static knowledge base (RAG documents) ---

    def create_documents_table(self):
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY,
                content TEXT,
                embedding BLOB,
                source_file TEXT
            )
        """)
        self.conn.commit()

    def save_document(self, content, embedding, source_file=None):
        self.cursor.execute(
            "INSERT INTO documents (content, embedding, source_file) VALUES (?, ?, ?)",
            (content, embedding.tobytes(), source_file)
        )
        self.conn.commit()
        return self.cursor.lastrowid

    def forget_document(self, source_file):
        """Removes a document's chunks and every cached answer built from it.

        Cache rows with no recorded source are removed too. They were written before
        answers tracked their provenance, so there is no way to tell whether they came
        from this document - and serving a possibly-stale answer is worse than paying
        for one more LLM call."""
        self.cursor.execute("DELETE FROM documents WHERE source_file = ?", (source_file,))
        chunks_removed = self.cursor.rowcount
        self.cursor.execute(
            "DELETE FROM cache WHERE source_files IS NULL "
            "OR source_files = ? OR source_files LIKE ? OR source_files LIKE ? OR source_files LIKE ?",
            (source_file, f"{source_file},%", f"%,{source_file}", f"%,{source_file},%"))
        answers_removed = self.cursor.rowcount
        self.conn.commit()
        self._matrix_cache = None  # cache rows changed underneath it
        return chunks_removed, answers_removed

    def record_document(self, source_file, content_hash):
        self.cursor.execute(
            "INSERT OR REPLACE INTO documents_meta (source_file, content_hash, ingested_at) VALUES (?, ?, ?)",
            (source_file, content_hash, time.time()))
        self.conn.commit()

    def document_hashes(self):
        self.cursor.execute("SELECT source_file, content_hash FROM documents_meta")
        return dict(self.cursor.fetchall())

    def get_document(self, doc_id):
        self.cursor.execute(
            "SELECT id, content, embedding, source_file FROM documents WHERE id = ?",
            (doc_id,)
        )
        row = self.cursor.fetchone()
        if row is None:
            return None
        row_id, content, embedding_blob, source_file = row
        return {
            "id": row_id,
            "content": content,
            "embedding": np.frombuffer(embedding_blob, dtype=np.float32),
            "source_file": source_file,
        }

    def retrieve_top_documents(self, query_vector, top_k=2, source_file=None, restrict_to_top_source=True):
        """Cosine-similarity search over the documents table; returns the top_k
        (similarity, content, source_file) tuples, highest similarity first.

        source_file: if given, searches ONLY chunks from that source (explicit
        override - useful for isolating one document during testing/debugging).

        restrict_to_top_source: if True (default) and source_file is None, first finds
        the single best-matching CHUNK across every source, then restricts the top_k
        search to only that chunk's source_file. This is the fix for cross-document
        contamination: with multiple documents pooled in one table, a wide top_k on a
        question about one document could pull in weakly-related chunks from an
        unrelated one - confirmed on a real run, where an OSHA-domain query's top_k=4
        included an ACME-manual chunk at 0.4641 similarity alongside OSHA chunks
        scoring 0.73/0.51/0.50, and this pattern is also the leading hypothesis for a
        282s runaway-generation incident (see project notes). Assumes a question is
        about ONE document at a time; pass False to restore the old
        pooled-across-all-sources behavior if that assumption stops holding (e.g. once
        genuinely cross-document questions are part of the test set).

        Vectorized computation - see _batch_cosine_similarity. Uses
        np.argsort(..., kind='stable') so that ties resolve in the exact same order as
        the old (Python stable-sort-based) implementation."""
        if source_file is not None:
            self.cursor.execute("SELECT content, embedding, source_file FROM documents WHERE source_file = ?", (source_file,))
        else:
            self.cursor.execute("SELECT content, embedding, source_file FROM documents")
        rows = self.cursor.fetchall()
        if not rows:
            return []
        contents = [r[0] for r in rows]
        source_files = [r[2] for r in rows]
        stored_matrix = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
        similarities = _batch_cosine_similarity(query_vector, stored_matrix)

        if source_file is None and restrict_to_top_source:
            winning_source = source_files[int(np.argmax(similarities))]
            keep = [i for i, sf in enumerate(source_files) if sf == winning_source]
            contents = [contents[i] for i in keep]
            source_files = [source_files[i] for i in keep]
            similarities = similarities[keep]

        order = np.argsort(-similarities, kind="stable")[:top_k]
        return [(float(similarities[i]), contents[i], source_files[i]) for i in order]

    def close(self):
        self.conn.close()


if __name__ == "__main__":
    if "--prewarm" in sys.argv:
        # Off-peak pre-warm run. In a real deployment, this would be
        # triggered by a scheduled task at night, e.g. (cron):
        #   0 3 * * * /usr/bin/python3 /path/to/gptcache_pipeline.py --prewarm
        # A separate branch so it doesn't mix with the live/reactive demo block below.
        cache = LocalSemanticCache()
        candidates = prioritize_candidates(generate_candidate_combinations())
        print(f"[PREWARM] {len(candidates)} candidate combinations, processing in priority order...")
        stats = prewarm_cache(cache, candidates)
        print(f"[PREWARM] Done: {stats}")
        print("Cache stats:", cache.stats())
        cache.close()
        sys.exit(0)

    if "--chat" in sys.argv:
        # Interactive assistant. Every answer goes through the same ask() path the
        # benchmark measures, so what is demonstrated here is exactly what was measured.
        cache = LocalSemanticCache()
        ingest_folder(cache, verbose=False)
        cache.cursor.execute("SELECT COUNT(*), COUNT(DISTINCT source_file) FROM documents")
        chunk_count, source_count = cache.cursor.fetchone()

        print("Electrical engineering technical support assistant.")
        print(f"Answering from {chunk_count} passages across {source_count} document(s), fully offline.")
        print("Type 'sources' to list the documents, 'stats' for cache statistics, 'exit' to quit.\n")

        while True:
            try:
                question = input("Question: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if question.lower() in {"exit", "quit", "q", ""}:
                break
            if question.lower() == "sources":
                cache.cursor.execute(
                    "SELECT source_file, COUNT(*) FROM documents GROUP BY source_file ORDER BY source_file")
                for source_file, n in cache.cursor.fetchall():
                    print(f"  {source_file} ({n} passages)")
                print()
                continue
            if question.lower() == "stats":
                print(f"  {cache.stats()}\n")
                continue

            answer, was_hit = ask(cache, question, verbose=False)
            timings = dict(LAST_TIMINGS)
            elapsed = timings.get("total", 0.0)
            if was_hit:
                # Showing this is the point of the project: the same question asked a
                # second time costs no LLM call at all.
                origin = f"answered from cache in {elapsed:.2f}s, similarity {timings.get('similarity', 0) * 100:.0f}%"
            else:
                sources = ", ".join(timings.get("sources", [])) or "no matching document"
                origin = f"read {timings.get('chunks_used', 0)} passage(s) from {sources}, {elapsed:.1f}s"
            print(f"[{origin}]")
            print(f"Answer:{answer}\n")

        cache.close()
        sys.exit(0)

    # --- chunk_text check: a 1000-word sample text ---
    sample_note = " ".join(f"word{i}" for i in range(1, 1001))  # a 1000-word fake physics note
    chunks = chunk_text(sample_note, chunk_size=500, overlap=50)
    print(f"[CHUNK CHECK] {len(sample_note.split())}-word text split into {len(chunks)} chunks:")
    for i, c in enumerate(chunks, start=1):
        words_in_chunk = c.split()
        print(f"  Chunk {i}: {len(words_in_chunk)} words | starts: '{words_in_chunk[0]}' | ends: '{words_in_chunk[-1]}'")

    cache = LocalSemanticCache()

    # --- [EDUCATION-DOMAIN TESTS - disabled during EE pivot, kept for later reference] ---
#     # --- Domain-focused cache tests (oversea education advising) ---
#     # Two groups, each following the same pattern: (a) a paraphrase pair, same
#     # country/exam -> CACHE HIT expected; (b) structurally identical but a different
#     # country -> CACHE MISS expected thanks to the entities_agree guard (the domain
#     # version of the earlier Turkey/Bulgaria case).
#     test_prompts = [
#         "What is the minimum IELTS score required for a master's degree in the UK?",
#         "How much IELTS do I need for a UK master's program?",                        # paraphrase -> HIT expected
#         "What is the minimum IELTS score required for a master's degree in Germany?",  # country changed -> MISS expected
#         "What documents do I need for a US student visa application?",
#         "What paperwork is required for an American student visa?",                   # paraphrase -> HIT expected
#         "What documents do I need for a Canadian student visa application?",           # country changed -> MISS expected
#     ]

#     for prompt in test_prompts:
#         start = time.time()
#         canonical_query = canonicalize_query(prompt)
#         query_vector = get_real_embedding(canonical_query)
#         match = cache.search(query_vector, raw_query=prompt, threshold=0.55)  # TODO: can be tuned against real scores - see the [DEBUG] lines

#         print(f"Original prompt : {prompt}")
#         print(f"Canonical query : {canonical_query}")

#         # For calibration: the nearest cache record's score, independent of the threshold
#         if VERBOSE_DIAGNOSTICS:
#             diag = cache.best_match_score(query_vector)
#             if diag is not None:
#                 print(f"[DEBUG] Nearest cache record: {diag[1] * 100:.2f}% -> '{diag[0]}'")

#         if match:
#             question, answer, similarity = match
#             elapsed = time.time() - start
#             print(f"[CACHE HIT] Similarity: {similarity * 100:.2f}% | Time: {elapsed:.4f}s")
#             print(f"Matched question: {question}")
#             print(f"Answer: {answer}\n")
#         else:
#             print("[CACHE MISS] Querying local LLM...")
#             answer = get_llm_response(prompt)  # generate the answer with the natural/original question
#             elapsed = time.time() - start
#             cache.save(canonical_query, answer, query_vector, raw_prompt=prompt)  # but cache the standardized form
#             print(f"[LLM RESPONSE] Time: {elapsed:.4f}s")
#             print(f"Answer: {answer}\n")

#     # --- Domain scope test ---
#     # An off-domain (physics) question is now subject to DOMAIN_SYSTEM_PROMPT; check
#     # whether the LLM declines to answer it (or states it's outside its domain).
#     off_domain_answer = get_llm_response("Can you explain Kirchhoff's laws to me?")
#     print(f"[DOMAIN SCOPE CHECK] Answer to an off-domain question: {off_domain_answer}\n")

#     # --- Documents-table check: write a domain note, read it back as a BLOB ---
#     test_vector = get_real_embedding("Standard IELTS requirement for UK master's programs")
#     doc_id = cache.save_document(
#         content="For most UK master's programs, the standard minimum IELTS score is 6.5 overall, with no band below 6.0.",
#         embedding=test_vector,
#         source_file="oversea_education_notes_general.txt"
#     )
#     retrieved = cache.get_document(doc_id)
#     if retrieved is not None and np.allclose(retrieved["embedding"], test_vector):
#         print(f"[DOCUMENTS CHECK OK] documents table works - id={doc_id}, source={retrieved['source_file']}")
#     else:
#         print("[DOCUMENTS CHECK FAIL] the vector read from the documents table doesn't match what was saved")

#     # --- RAG check: retrieval + prompt integration, domain version ---
#     # Deliberately uses a made-up-but-realistic institution-specific rule that a
#     # general LLM couldn't know. If this detail (SOAP + the 6.5 exception) shows up
#     # correctly in the answer, that proves it came from retrieval, not general
#     # knowledge.
#     custom_exchange_rule_note = (
#         "According to XYZ University's Fall 2026 Exchange Program guide, applicants "
#         "coming through a partner-institution exchange track must submit a 'Statement "
#         "of Academic Purpose' (SOAP) no later than 45 days before the semester start. "
#         "For this specific exchange track only, the minimum accepted IELTS score is "
#         "6.5, which is lower than the standard 7.0 required for regular (non-exchange) "
#         "applicants to the same program."
#     )
#     note_vector = get_real_embedding(custom_exchange_rule_note)
#     cache.save_document(
#         content=custom_exchange_rule_note,
#         embedding=note_vector,
#         source_file="oversea_education_notes_xyz_exchange.txt"
#     )

#     rag_query = "What IELTS score do exchange-track applicants need for XYZ University's Fall 2026 program, and what document must they submit?"
#     rag_answer, top_docs = answer_with_rag(cache, rag_query, top_k=2)

#     print(f"[RAG CHECK] RAG query: {rag_query}")
#     for score, content, source_file in top_docs:
#         print(f"  Chunk used (score={score:.4f}, source={source_file}): {content[:80]}...")
#     print(f"[RAG CHECK] LLM answer: {rag_answer}")

    # --- Load every document in the folder (Step 7 of the EE pivot) ---
    # Replaces three hand-written ingestion blocks that each carried their own file path,
    # chunk size and overlap. Chunk size is now measured per document (see
    # pick_chunk_size), and adding a document is a matter of dropping a .txt file into
    # ee_documents/ - including the synthetic ACME manual, which used to be a string
    # inside this file and is now just another document in the folder.
    ingest_folder(cache)

    source_scope_query = "What is the minimum headroom for working space about service equipment installed on or after August 13, 2007?"
    q_vec = get_real_embedding(source_scope_query)
    pooled = cache.retrieve_top_documents(q_vec, top_k=4, restrict_to_top_source=False)
    scoped = cache.retrieve_top_documents(q_vec, top_k=4, restrict_to_top_source=True)
    pooled_sources = {sf for _, _, sf in pooled}
    scoped_sources = {sf for _, _, sf in scoped}
    print(f"[SOURCE SCOPE CHECK] Query: {source_scope_query}")
    print(f"  Pooled (old behavior)   top_k=4 sources: {pooled_sources}")
    print(f"  Scoped (new default)    top_k=4 sources: {scoped_sources}")
    print(f"  {'OK - scoped result stayed within one source' if len(scoped_sources) <= 1 else 'UNEXPECTED - scoped result still mixed sources'}")

    real_doc_queries = [
        "For a 15-ampere receptacle on a 15- or 20-ampere branch circuit supplying two or more receptacles, what is the maximum cord- and plug-connected load allowed?",
        "By what factor may the continuous ampere rating of a fuse exceed the conductor ampacity, and what is the maximum for a breaker's long-time trip setting?",
    ]
    for q in real_doc_queries:
        real_doc_answer, real_doc_top_docs = answer_with_rag(cache, q, top_k=4)
        print(f"[REAL DOC RAG CHECK] Query: {q}")
        for score, content, source_file in real_doc_top_docs:
            print(f"  Chunk used (score={score:.4f}, source={source_file}): {content[:80]}...")
        print(f"[REAL DOC RAG CHECK] LLM answer: {real_doc_answer}")

    # --- [ask()<->RAG integration check - superseded by run_benchmark() below,
    #      which exercises the same path with a larger query set] ---
#     # --- ask() <-> RAG integration check (Step 2 of the EE pivot) ---
#     # Same questions as above, but now going through ask() instead of calling
#     # answer_with_rag() directly - exercises the fast-path/canonical-path cache checks
#     # AND the new RAG-backed miss path, plus caching of the RAG answer itself.
#     integration_queries = [
#         ee_rag_query,             # ACME question
#         real_doc_queries[0],      # OSHA Table S-4 question
#         real_doc_queries[1],      # OSHA fuse/breaker overcurrent question
#     ]
#     for q in integration_queries:
#         answer, was_hit = ask(cache, q)
#         print(f"[ASK+RAG CHECK] Query: {q}")
#         print(f"[ASK+RAG CHECK] was_cache_hit={was_hit} | Answer: {answer}")

#     # Ask the SAME questions again - this time they should all be fast-path cache hits
#     print("[ASK+RAG CHECK] Re-asking the same 3 questions - expecting cache hits this time:")
#     for q in integration_queries:
#         answer, was_hit = ask(cache, q)
#         print(f"[ASK+RAG CHECK] was_cache_hit={was_hit} | Query: {q[:60]}...")

    # --- [Prompt experiment - answered its question on 10 September, left here so the
    #      result can be reproduced. Across 20 trials (4 instruction variants x 5) the
    #      ACME question was wrong every single time, including with an explicit
    #      "the provided information overrides your own knowledge" line in the system
    #      message, next to the context, and in both. The model's prior knowledge wins
    #      regardless of instruction, so this is a model limitation, not a prompt or a
    #      retrieval problem. Uncomment to re-run.] ---
    # acme_spec = next(s for s in BENCHMARK_QUERIES if s["q"].startswith("What type of circuit breaker curve"))
    # run_prompt_experiment(cache, acme_spec, trials=5)
    # sys.exit(0)

    # --- Measurement run ---
    # New label: automatic chunk sizing changes the chunk boundaries, so these rows are
    # NOT comparable with the ee_final_topk8 rows and must not be averaged with them.
    # The final measurement, run on the frozen code. The 85.3% result was measured
    # before document change detection and the rebuilt chat mode reached the machine;
    # neither should affect the benchmark, but the figures quoted in the report need to
    # come from the exact code that was submitted, so this run is the one to cite.
    run_benchmark(cache, label="ee_final_frozen", repeats=5)

    # --- [Diagnostic experiments - disabled for the final measurement run.
    #      They answered their questions (Q13 = recall, rank 5 outside top_k=4;
    #      Q8 = reproducible prompt sensitivity) and each costs dozens of extra
    #      LLM calls. Re-enable by uncommenting if a new question comes up.] ---
#     # --- Controlled A/B on the ACME 7 kW question (Step 5 of the EE pivot) ---
#     # This question scored 3/3 correct under pooled retrieval and 0/3 under
#     # source-scoped retrieval. That looks causal, but the two runs differ by only one
#     # unrelated chunk and 3 trials is far too few to tell a real effect from noise at
#     # temperature 0.3, so it is tested directly here instead of being assumed.
#     acme_spec = next(s for s in BENCHMARK_QUERIES if s["q"].startswith("What type of circuit breaker curve"))
#     run_ab_experiment(
#         cache, acme_spec,
#         variants=[
#             ("scoped top_k=4 (current default)", {"top_k": 4, "restrict_to_top_source": True}),
#             ("pooled top_k=4 (previous default)", {"top_k": 4, "restrict_to_top_source": False}),
#             ("scoped top_k=3 (all ACME chunks)", {"top_k": 3, "restrict_to_top_source": True}),
#         ],
#         trials=5,
#     )

#     # --- top_k sweep on the two table-lookup failures (Step 6 of the EE pivot) ---
#     # Q13 and Q14 are both 0/3, but they fail DIFFERENTLY: Q13 answered from the wrong
#     # paragraph (it gave the over-600V figure for a 600V-or-less question), while Q14
#     # declined outright, meaning its chunk never reached the context at all. Both point
#     # at recall - with 30 chunks in 1910.303, top_k=4 may simply be too narrow for
#     # table-based facts. Testing that directly instead of assuming it.
#     for q_prefix, needle in (("What is the minimum width of working space", "762 mm"),
#                             ("For equipment between 9001 V and 25 kV", "9001 V-25 kV")):
#         spec = next(s for s in BENCHMARK_QUERIES if s["q"].startswith(q_prefix))
#         rank = report_chunk_rank(cache, spec["q"], needle)
#         # Sweep beyond the measured rank, otherwise the variants cannot include the
#         # answer chunk and the experiment tests nothing (which is what happened on the
#         # previous run, where this query's chunk ranked 13 and the sweep stopped at 12).
#         sweep = [4, 8, 12]
#         if rank is not None and rank > max(sweep):
#             sweep += [rank + 2, rank + 8]
#         run_ab_experiment(
#             cache, spec,
#             variants=[(f"scoped top_k={k}", {"top_k": k, "restrict_to_top_source": True}) for k in sweep],
#             trials=3,
#             answer_rank=rank,
#         )

    print("Cache stats:", cache.stats())
    cache.close()
