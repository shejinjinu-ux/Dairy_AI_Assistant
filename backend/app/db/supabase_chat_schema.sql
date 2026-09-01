-- ====================================================================
-- Dairy AI Assistant - Supabase / PostgreSQL Production Schema
-- Global Tag ID Uniqueness, Persistent Milk History, Vaccinations, Chat & RLS
-- ====================================================================

-- 1. Farms Table
CREATE TABLE IF NOT EXISTS public.farms (
    farm_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    farm_name TEXT NOT NULL,
    location TEXT,
    is_demo BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 2. Cattle Table (Globally Unique Tag ID)
CREATE TABLE IF NOT EXISTS public.cattle (
    animal_id TEXT PRIMARY KEY,
    tag_id TEXT UNIQUE NOT NULL, -- Global Uniqueness Enforced
    farm_id TEXT REFERENCES public.farms(farm_id) ON DELETE SET NULL,
    user_id TEXT NOT NULL,
    name TEXT,
    species VARCHAR(30) NOT NULL DEFAULT 'Cattle',
    breed VARCHAR(50) NOT NULL DEFAULT 'Crossbred',
    gender VARCHAR(10) NOT NULL DEFAULT 'Female',
    age_months NUMERIC(6, 2),
    date_of_birth DATE,
    body_weight_kg NUMERIC(6, 2) NOT NULL DEFAULT 400.0,
    calving_date DATE,
    lactation_start_date DATE,
    parity INT NOT NULL DEFAULT 1,
    current_lactation_status VARCHAR(20) NOT NULL DEFAULT 'Lactating',
    days_in_milk NUMERIC(6, 2),
    lactation_stage VARCHAR(20),
    daily_milk_yield_litres NUMERIC(6, 2) NOT NULL DEFAULT 0.0,
    milk_fat_percentage NUMERIC(4, 2) NOT NULL DEFAULT 4.0,
    pregnancy_status BOOLEAN NOT NULL DEFAULT FALSE,
    pregnancy_month INT,
    is_demo BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 3. Persistent Milk Records Table
CREATE TABLE IF NOT EXISTS public.milk_records (
    record_id TEXT PRIMARY KEY,
    tag_id TEXT NOT NULL REFERENCES public.cattle(tag_id) ON DELETE CASCADE,
    user_id TEXT NOT NULL,
    farm_id TEXT,
    date DATE NOT NULL,
    morning_yield_litres NUMERIC(6, 2) NOT NULL DEFAULT 0.0,
    evening_yield_litres NUMERIC(6, 2) NOT NULL DEFAULT 0.0,
    total_yield_litres NUMERIC(6, 2) NOT NULL,
    fat_percentage NUMERIC(4, 2),
    snf_percentage NUMERIC(4, 2),
    notes TEXT,
    is_demo BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 4. Persistent Vaccination Records Table
CREATE TABLE IF NOT EXISTS public.vaccination_records (
    record_id TEXT PRIMARY KEY,
    tag_id TEXT NOT NULL REFERENCES public.cattle(tag_id) ON DELETE CASCADE,
    user_id TEXT NOT NULL,
    disease_target VARCHAR(50) NOT NULL,
    vaccine_name TEXT NOT NULL,
    administered_date DATE NOT NULL,
    next_due_date DATE NOT NULL,
    recommended_timing TEXT NOT NULL,
    status VARCHAR(30) NOT NULL DEFAULT 'COMPLETED',
    estimated_cost_inr NUMERIC(8, 2),
    batch_number TEXT,
    veterinarian_name TEXT,
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 5. Chat Sessions Table
CREATE TABLE IF NOT EXISTS public.chat_sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT,
    language VARCHAR(10) NOT NULL DEFAULT 'en',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 6. Chat Messages Table
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

-- 7. Performance Indexes
CREATE INDEX IF NOT EXISTS idx_cattle_tag_id ON public.cattle(tag_id);
CREATE INDEX IF NOT EXISTS idx_cattle_user_id ON public.cattle(user_id);
CREATE INDEX IF NOT EXISTS idx_milk_records_tag_id ON public.milk_records(tag_id);
CREATE INDEX IF NOT EXISTS idx_milk_records_date ON public.milk_records(date DESC);
CREATE INDEX IF NOT EXISTS idx_vaccination_records_tag_id ON public.vaccination_records(tag_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id ON public.chat_messages(session_id);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_id ON public.chat_sessions(user_id);

-- 8. Row Level Security (RLS)
ALTER TABLE public.farms ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.cattle ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.milk_records ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.vaccination_records ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.chat_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.chat_messages ENABLE ROW LEVEL SECURITY;

-- 9. Tenant Isolation Policies
CREATE POLICY "Users can manage own cattle" ON public.cattle
    FOR ALL USING (auth.uid()::text = user_id OR user_id IS NULL)
    WITH CHECK (auth.uid()::text = user_id OR user_id IS NULL);

CREATE POLICY "Users can manage own milk records" ON public.milk_records
    FOR ALL USING (auth.uid()::text = user_id OR user_id IS NULL)
    WITH CHECK (auth.uid()::text = user_id OR user_id IS NULL);

CREATE POLICY "Users can manage own vaccination records" ON public.vaccination_records
    FOR ALL USING (auth.uid()::text = user_id OR user_id IS NULL)
    WITH CHECK (auth.uid()::text = user_id OR user_id IS NULL);
