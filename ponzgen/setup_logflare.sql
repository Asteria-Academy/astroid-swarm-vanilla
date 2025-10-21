-- Enable Wrappers extension
CREATE EXTENSION IF NOT EXISTS wrappers WITH SCHEMA extensions;

-- Enable Logflare wrapper
CREATE FOREIGN DATA WRAPPER logflare_wrapper
  HANDLER logflare_fdw_handler
  VALIDATOR logflare_fdw_validator;

-- Create server with proper credentials
CREATE SERVER logflare_server
FOREIGN DATA WRAPPER logflare_wrapper
OPTIONS (
    api_key 'KHWKiahyAbNY',
    endpoint '7021a4e1-6046-4a06-8f8a-2af97024cee8'
);

-- Create schema for Logflare tables
CREATE SCHEMA IF NOT EXISTS logflare;

-- Create foreign table for log events
CREATE FOREIGN TABLE logflare.my_logflare_table (
    id bigint,
    event_message text,
    metadata jsonb
)
SERVER logflare_server
OPTIONS (
    endpoint '7021a4e1-6046-4a06-8f8a-2af97024cee8'
);

-- Create RLS policies
ALTER TABLE logflare.my_logflare_table ENABLE ROW LEVEL SECURITY;

-- Allow authenticated users to read
CREATE POLICY "Users can view logs"
    ON logflare.my_logflare_table
    FOR SELECT
    USING (true);
