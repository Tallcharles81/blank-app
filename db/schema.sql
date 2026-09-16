-- Schema for the NFL DFS optimizer database.
-- Applied via db/migrate.py:run_migrations(), or manually with:
--   psql "$DATABASE_URL" -f db/schema.sql

CREATE TABLE IF NOT EXISTS player_weekly_stats (
    player_id           TEXT NOT NULL,
    player_name         TEXT NOT NULL,
    position            TEXT NOT NULL,
    team                TEXT NOT NULL,
    season              INTEGER NOT NULL,
    week                INTEGER NOT NULL,
    targets             INTEGER,
    target_share        NUMERIC(5, 4),
    air_yards_share     NUMERIC(5, 4),
    red_zone_targets    INTEGER,
    carries             INTEGER,
    red_zone_carries    INTEGER,
    snap_pct            NUMERIC(5, 4),
    rushing_yards       NUMERIC(6, 2),
    receiving_yards     NUMERIC(6, 2),
    fantasy_points_ppr  NUMERIC(6, 2),
    opponent            TEXT,
    vegas_implied_total NUMERIC(5, 2),
    PRIMARY KEY (player_id, season, week)
);

-- Added for the nflverse injuries fetch (data/nflverse_fetch.py) - the official
-- weekly report status (e.g. "Out", "Questionable"), not the full injury detail.
ALTER TABLE player_weekly_stats ADD COLUMN IF NOT EXISTS injury_status TEXT;

CREATE INDEX IF NOT EXISTS idx_player_weekly_stats_season_week
    ON player_weekly_stats (season, week);

CREATE TABLE IF NOT EXISTS slate_player_pool (
    slate_id             TEXT NOT NULL,
    player_id            TEXT NOT NULL,
    name                 TEXT NOT NULL,
    position             TEXT NOT NULL,
    salary               INTEGER NOT NULL,
    team                 TEXT NOT NULL,
    opponent             TEXT,
    game_time            TIMESTAMPTZ,
    avg_points_per_game  NUMERIC(6, 2),
    PRIMARY KEY (slate_id, player_id)
);

CREATE INDEX IF NOT EXISTS idx_slate_player_pool_slate_id
    ON slate_player_pool (slate_id);

CREATE TABLE IF NOT EXISTS projections (
    slate_id           TEXT NOT NULL,
    player_id          TEXT NOT NULL,
    proj_floor         NUMERIC(6, 2),
    proj_median        NUMERIC(6, 2),
    proj_ceiling       NUMERIC(6, 2),
    proj_percentiles   JSONB,
    locked_at          TIMESTAMPTZ,
    PRIMARY KEY (slate_id, player_id),
    FOREIGN KEY (slate_id, player_id)
        REFERENCES slate_player_pool (slate_id, player_id)
);

-- locked_at is set once and must never change afterwards.
CREATE OR REPLACE FUNCTION prevent_locked_at_update() RETURNS TRIGGER AS $$
BEGIN
    IF OLD.locked_at IS NOT NULL AND NEW.locked_at IS DISTINCT FROM OLD.locked_at THEN
        RAISE EXCEPTION 'locked_at cannot be modified once set (slate_id=%, player_id=%)',
            OLD.slate_id, OLD.player_id;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS projections_lock_immutable ON projections;
CREATE TRIGGER projections_lock_immutable
    BEFORE UPDATE ON projections
    FOR EACH ROW
    EXECUTE FUNCTION prevent_locked_at_update();

CREATE TABLE IF NOT EXISTS actuals (
    id                  BIGSERIAL PRIMARY KEY,
    slate_id            TEXT NOT NULL,
    player_id           TEXT NOT NULL,
    actual_score        NUMERIC(6, 2),
    contest_id          TEXT,
    lineup_id           TEXT,
    win_prob_at_lock    NUMERIC(5, 4),
    result              TEXT,
    FOREIGN KEY (slate_id, player_id)
        REFERENCES slate_player_pool (slate_id, player_id),
    UNIQUE (slate_id, player_id, contest_id, lineup_id)
);

CREATE INDEX IF NOT EXISTS idx_actuals_slate_id
    ON actuals (slate_id);

CREATE INDEX IF NOT EXISTS idx_actuals_contest_id
    ON actuals (contest_id);

-- One row per backtested (season, week) - see models/calibration.py. Stored
-- per-position/per-week counts (not just an already-averaged MAE) so weeks
-- can be pooled correctly afterward: sum(mae * n) / sum(n), not a naive
-- average-of-per-week-averages that would equal-weight a 5-player week the
-- same as a 300-player week. Same reasoning for p80_hits/p80_opportunities.
CREATE TABLE IF NOT EXISTS calibration_weekly (
    season                  INTEGER NOT NULL,
    week                    INTEGER NOT NULL,
    num_players_evaluated   INTEGER NOT NULL,
    mae_by_position         JSONB,
    p20_hits                INTEGER,
    p20_opportunities       INTEGER,
    p50_hits                INTEGER,
    p50_opportunities       INTEGER,
    p80_hits                INTEGER,
    p80_opportunities       INTEGER,
    avg_field_percentile    NUMERIC(5, 4),
    field_size              INTEGER,
    computed_at             TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (season, week)
);

-- Added after the table already existed with p80-only data - CREATE TABLE IF
-- NOT EXISTS above doesn't add columns to an existing table.
ALTER TABLE calibration_weekly ADD COLUMN IF NOT EXISTS p20_hits INTEGER;
ALTER TABLE calibration_weekly ADD COLUMN IF NOT EXISTS p20_opportunities INTEGER;
ALTER TABLE calibration_weekly ADD COLUMN IF NOT EXISTS p50_hits INTEGER;
ALTER TABLE calibration_weekly ADD COLUMN IF NOT EXISTS p50_opportunities INTEGER;

-- Team-level pass defense performance, aggregated from PFR's real per-defender
-- weekly coverage charting (see data/nflverse_fetch.py:fetch_pfr_def_advstats
-- and models/matchups.py for what this is and, just as importantly, what real
-- nflverse data does NOT support - per-play defender assignment and
-- slot/wide receiver alignment are both confirmed absent). Summed across all
-- of a team's charted defenders in a week, since a live DK slate can't know
-- in advance which specific defender covers which specific receiver.
CREATE TABLE IF NOT EXISTS team_pass_defense_weekly (
    team                        TEXT NOT NULL,
    season                      INTEGER NOT NULL,
    week                        INTEGER NOT NULL,
    def_targets                 INTEGER,
    def_completions_allowed     INTEGER,
    def_yards_allowed           NUMERIC(7, 2),
    def_receiving_td_allowed    INTEGER,
    PRIMARY KEY (team, season, week)
);

-- Durable record of pre-lock web-research checks (see data/pre_lock_check.py)
-- - a second, independent check of injury news/inactive lists/weather in the
-- hours before a slate locks, specifically for news that broke faster than
-- the roster/injury data feeds (see data/player_availability.py) caught up
-- to. This table is what makes a check's result auditable after the fact,
-- not just a chat message that scrolls away - `contradicts_gate` is what a
-- caller queries to find something that needs loud attention right now.
CREATE TABLE IF NOT EXISTS pre_lock_checks (
    id                  BIGSERIAL PRIMARY KEY,
    slate_id            TEXT NOT NULL,
    player_id           TEXT NOT NULL,
    source              TEXT NOT NULL,
    finding             TEXT NOT NULL,
    gate_status_at_check TEXT,
    contradicts_gate    BOOLEAN NOT NULL DEFAULT FALSE,
    checked_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    FOREIGN KEY (slate_id, player_id)
        REFERENCES slate_player_pool (slate_id, player_id)
);

CREATE INDEX IF NOT EXISTS idx_pre_lock_checks_slate_id
    ON pre_lock_checks (slate_id);
