-- 1. Roles Enum
CREATE TYPE user_roles AS ENUM ('admin', 'user');

-- 2. Users Table
CREATE TABLE users (
    id TEXT PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    role user_roles DEFAULT 'user' NOT NULL,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL
);

CREATE INDEX idx_users_email ON users(email);

-- 3. Documents Table (File Registry)
CREATE TABLE documents (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    is_public BOOLEAN DEFAULT FALSE NOT NULL,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL
);

CREATE INDEX idx_documents_user_id ON documents(user_id);
CREATE INDEX idx_documents_is_public ON documents(is_public);

-- 4. Chunks Table (Isolated Keyword & Full-Text Store)
CREATE TABLE chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    is_public BOOLEAN DEFAULT FALSE NOT NULL,
    text TEXT NOT NULL,
    source TEXT NOT NULL,
    search_vector TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
);

CREATE INDEX idx_chunks_document_id ON chunks(document_id);
CREATE INDEX idx_chunks_user_id ON chunks(user_id);
CREATE INDEX idx_chunks_is_public ON chunks(is_public);
CREATE INDEX idx_search_vector ON chunks USING GIN(search_vector);