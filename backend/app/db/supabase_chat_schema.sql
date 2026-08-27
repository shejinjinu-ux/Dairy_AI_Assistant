-- ====================================================================
-- Dairy AI Assistant - Supabase / PostgreSQL Production Schema for Chat
-- Idempotent, High-Performance, and RLS-Secured
-- ====================================================================

-- 1. Create chat_sessions table
CREATE TABLE IF NOT EXISTS public.chat_sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT,
    language VARCHAR(10) NOT NULL DEFAULT 'en',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 2. Create chat_messages table
CREATE TABLE IF NOT EXISTS public.chat_messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES public.chat_sessions(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL, -- 'user', 'assistant', 'system'
    message TEXT NOT NULL,
    language VARCHAR(10) NOT NULL DEFAULT 'en',
    intent VARCHAR(50),
    module VARCHAR(50),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 3. Performance Indexes
CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id ON public.chat_messages(session_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_created_at ON public.chat_messages(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_id ON public.chat_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_updated_at ON public.chat_sessions(updated_at DESC);

-- 4. Enable Row Level Security (RLS)
ALTER TABLE public.chat_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.chat_messages ENABLE ROW LEVEL SECURITY;

-- 5. Row Level Security Policies for chat_sessions
-- Idempotent drop if exists
DROP POLICY IF EXISTS "Service role has full access to chat_sessions" ON public.chat_sessions;
DROP POLICY IF EXISTS "Users can read own chat sessions" ON public.chat_sessions;
DROP POLICY IF EXISTS "Users can insert own chat sessions" ON public.chat_sessions;
DROP POLICY IF EXISTS "Users can update own chat sessions" ON public.chat_sessions;
DROP POLICY IF EXISTS "Users can delete own chat sessions" ON public.chat_sessions;

-- Allow service role full unrestricted access
CREATE POLICY "Service role has full access to chat_sessions"
    ON public.chat_sessions
    FOR ALL
    USING (auth.role() = 'service_role')
    WITH CHECK (auth.role() = 'service_role');

-- Allow users / anon to read their own or anonymous sessions
CREATE POLICY "Users can read own chat sessions"
    ON public.chat_sessions
    FOR SELECT
    USING (auth.uid()::text = user_id OR user_id IS NULL);

-- Allow users / anon to insert sessions
CREATE POLICY "Users can insert own chat sessions"
    ON public.chat_sessions
    FOR INSERT
    WITH CHECK (auth.uid()::text = user_id OR user_id IS NULL);

-- Allow users / anon to update their own sessions
CREATE POLICY "Users can update own chat sessions"
    ON public.chat_sessions
    FOR UPDATE
    USING (auth.uid()::text = user_id OR user_id IS NULL)
    WITH CHECK (auth.uid()::text = user_id OR user_id IS NULL);

-- Allow users / anon to delete their own sessions
CREATE POLICY "Users can delete own chat sessions"
    ON public.chat_sessions
    FOR DELETE
    USING (auth.uid()::text = user_id OR user_id IS NULL);


-- 6. Row Level Security Policies for chat_messages
-- Idempotent drop if exists
DROP POLICY IF EXISTS "Service role has full access to chat_messages" ON public.chat_messages;
DROP POLICY IF EXISTS "Users can read messages for accessible sessions" ON public.chat_messages;
DROP POLICY IF EXISTS "Users can insert messages into accessible sessions" ON public.chat_messages;
DROP POLICY IF EXISTS "Users can update messages in accessible sessions" ON public.chat_messages;
DROP POLICY IF EXISTS "Users can delete messages from accessible sessions" ON public.chat_messages;

-- Allow service role full unrestricted access
CREATE POLICY "Service role has full access to chat_messages"
    ON public.chat_messages
    FOR ALL
    USING (auth.role() = 'service_role')
    WITH CHECK (auth.role() = 'service_role');

-- Allow users to read messages for sessions they own or anonymous sessions
CREATE POLICY "Users can read messages for accessible sessions"
    ON public.chat_messages
    FOR SELECT
    USING (EXISTS (
        SELECT 1 FROM public.chat_sessions s
        WHERE s.id = chat_messages.session_id
        AND (s.user_id = auth.uid()::text OR s.user_id IS NULL)
    ));

-- Allow users to insert messages into accessible sessions
CREATE POLICY "Users can insert messages into accessible sessions"
    ON public.chat_messages
    FOR INSERT
    WITH CHECK (EXISTS (
        SELECT 1 FROM public.chat_sessions s
        WHERE s.id = chat_messages.session_id
        AND (s.user_id = auth.uid()::text OR s.user_id IS NULL)
    ));

-- Allow users to update messages in accessible sessions
CREATE POLICY "Users can update messages in accessible sessions"
    ON public.chat_messages
    FOR UPDATE
    USING (EXISTS (
        SELECT 1 FROM public.chat_sessions s
        WHERE s.id = chat_messages.session_id
        AND (s.user_id = auth.uid()::text OR s.user_id IS NULL)
    ))
    WITH CHECK (EXISTS (
        SELECT 1 FROM public.chat_sessions s
        WHERE s.id = chat_messages.session_id
        AND (s.user_id = auth.uid()::text OR s.user_id IS NULL)
    ));

-- Allow users to delete messages from accessible sessions
CREATE POLICY "Users can delete messages from accessible sessions"
    ON public.chat_messages
    FOR DELETE
    USING (EXISTS (
        SELECT 1 FROM public.chat_sessions s
        WHERE s.id = chat_messages.session_id
        AND (s.user_id = auth.uid()::text OR s.user_id IS NULL)
    ));
