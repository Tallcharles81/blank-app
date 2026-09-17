# 🏀 NBA DraftKings Lineup Optimizer

A Streamlit app that builds optimal DraftKings NBA Classic lineups from a
player-pool CSV. It uses integer linear programming (PuLP) to maximize
projected fantasy points subject to DraftKings' real contest rules:

- $50,000 salary cap
- 8 roster slots: PG, SG, SF, PF, C, G, F, UTIL, each respecting position
  eligibility (e.g. a PG/SG can fill PG, SG, or G)
- Players must come from at least 2 different games on the slate

### Features

- Upload a DraftKings CSV export, or use the bundled sample slate
  (`sample_data.csv`)
- Edit player projections directly in the table
- Lock players into every lineup, or exclude them entirely
- Generate multiple diverse lineups at once (for GPP-style multi-entry),
  with a configurable max overlap between lineups
- Download all generated lineups as a CSV

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```

### CSV format

The app accepts DraftKings' standard "Export Player List" columns
(`Position`, `Name`, `Salary`, `Game Info`, `TeamAbbrev`,
`AvgPointsPerGame`), or a simplified CSV with at least `Name`, `Salary`,
and `Position` columns. `Position` may contain multiple positions
separated by `/`, e.g. `PG/SG`.
