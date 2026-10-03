"""Settings for the whole project, in one place.

Right now (Lesson 1) we only need to know where the runbooks live and how big we
want our chunks to be. Later lessons will add database and API-key settings here.
"""

# The folder that holds your runbooks (the documents we search over).
CORPUS_DIR = "corpus"

# Fixed-size chunking: how big each chunk is, and how much neighbouring chunks
# overlap. Overlap means a sentence that sits on a boundary still appears whole
# in at least one chunk, so we don't cut ideas in half.
CHUNK_SIZE = 800      # characters
CHUNK_OVERLAP = 150   # characters

# Header chunking: if one section under a heading is bigger than this, we split
# it further so no single chunk is enormous.
MAX_SECTION_SIZE = 1200  # characters
