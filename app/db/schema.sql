-- 1. Users Table (role: 1 = USER, 2 = ADMIN)
CREATE TABLE users (
    id TEXT PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100),
    hashed_password VARCHAR(255) NOT NULL,
    role INTEGER DEFAULT 1 NOT NULL CHECK (role IN (1, 2)),
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL
);

CREATE INDEX idx_users_email ON users(email);

-- 2. Documents Table (File Registry)
CREATE TABLE documents (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    file_hash TEXT,
    chunk_count INTEGER DEFAULT 0 NOT NULL,
    user_id TEXT REFERENCES users(id) ON DELETE CASCADE,
    is_public BOOLEAN DEFAULT FALSE NOT NULL,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    CONSTRAINT chk_documents_ownership CHECK (
        (is_public = TRUE AND user_id IS NULL) OR
        (is_public = FALSE AND user_id IS NOT NULL)
    )
);

CREATE INDEX idx_documents_user_id ON documents(user_id);
CREATE INDEX idx_documents_is_public ON documents(is_public);
CREATE INDEX idx_documents_file_hash ON documents(file_hash);

-- 3. Chunks Table (Isolated Keyword & Full-Text Store)
CREATE TABLE chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER DEFAULT 0 NOT NULL,
    user_id TEXT REFERENCES users(id) ON DELETE CASCADE,
    is_public BOOLEAN DEFAULT FALSE NOT NULL,
    text TEXT NOT NULL,
    source TEXT NOT NULL,
    search_vector TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    CONSTRAINT chk_chunks_ownership CHECK (
        (is_public = TRUE AND user_id IS NULL) OR
        (is_public = FALSE AND user_id IS NOT NULL)
    )
);

CREATE INDEX idx_chunks_document_id ON chunks(document_id);
CREATE INDEX idx_chunks_user_id ON chunks(user_id);
CREATE INDEX idx_chunks_is_public ON chunks(is_public);
CREATE INDEX idx_search_vector ON chunks USING GIN(search_vector);