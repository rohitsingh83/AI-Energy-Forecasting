# Claims audit — make the project credible

The original brief included useful motivation and several unsupported statistics.
A professional student repository should not repeat uncertain numbers as facts.
This audit distinguishes verified context, unverified assertions and project evidence.

## External claims in the brief

| Original claim | Treatment in this project |
|---|---|
| “Google saved 40% of energy used for cooling using AI” | Qualify: DeepMind's July 20, 2016 account reports **up to 40% reduction in cooling energy**, not 40% of all facility electricity and not a result of this forecasting project. It described predictive models and a real cooling-control intervention. |
| “AI energy forecasting market will reach $60 billion by 2030” | Omitted. No verified definition/source for this specific market/value was established. AI energy, smart-grid and general forecasting markets are not interchangeable. |
| “Named companies are actively hiring smart-grid AI engineers” | Not asserted as current hiring. Current openings depend on date/location and must be checked on official careers pages. This project does not establish demand at any particular employer. |
| “70% of India's power loss is due to poor forecasting/mismanagement” | Omitted. Unsupported and conflates forecasting with technical, commercial and operational loss categories. |
| “AI+energy engineers earn 30–50% more” | Omitted. No reliable role/location/experience-adjusted evidence supplied. Do not promise a salary premium. |
| “US DOE funded $100m specifically in AI forecasting R&D” | Omitted. No specific verified program/source established here; broader AI/energy funding is not the same claim. |
| “Smart cities run on AI energy systems” | Use as a potential application, not a universal factual statement. Smart-city infrastructure has many components and maturity levels. |
| “Forecasting prevents blackouts / ensures net zero” | Reworded: forecasts can support planning. Capacity, grid operations, physical constraints and actual interventions determine outcomes. |

### Official context source

Richard Evans and Jim Gao, **DeepMind AI Reduces Google Data Centre Cooling Bill
by 40%**, July 20, 2016:
https://deepmind.google/blog/deepmind-ai-reduces-google-data-centre-cooling-bill-by-40/

This is a historical example of AI-assisted industrial energy optimization,
**not evidence of this project's business impact or current hiring**.

## Verified dataset facts

Official UCI page:
https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

It identifies 2,075,259 observations, minute-averaged active power in kW, one
household in Sceaux, December 2006–November 2010, creators Hebrail/Berard,
DOI 10.24432/C58K54 and a CC BY 4.0 license. This project downloaded the official
archive, recorded its SHA256, adapted it hourly and preserved attribution.
See [DATA_CARD.md](DATA_CARD.md) and [DATA_LICENSE.md](../DATA_LICENSE.md).

## What the project itself actually demonstrates

- A runnable modular Python pipeline and API.
- Synthetic data clearly labeled as generated; a separate real public benchmark.
- Chronological validation and test scores reproducible from saved predictions.
- A validation-selected MLP for synthetic data and RF for measured household data.
- Lower sampled test day-ahead RMSE than a weekly-naive baseline on these two runs.
- A virtual load-shifting cost calculation with explicit assumptions.
- Automated checks and actual application screenshots.

It does **not** demonstrate live deployment, guaranteed accuracy, production
readiness, measured electricity/bill/carbon savings, outages prevented, a company
client, a hiring offer or a salary premium.

## Honest wording examples

| Avoid | Use instead |
|---|---|
| “Deployed AI to real smart grids” | “Served historical-data forecasts through a local Flask API” |
| “Saved companies 30% electricity” | “Compared forecast-based hypothetical load-shifting costs under illustrative rates” |
| “Achieved 83% accuracy” | “Synthetic test one-step R² was 0.831; day-ahead RMSE was 6.888 kW” |
| “Predicts any hour from weekday alone” | “Forecasts the next contiguous hours using saved recent demand and calendar” |
| “Reduced carbon by shifting energy” | “Conserved energy; constant-intensity modeled carbon stayed unchanged” |
| “Used corporate smart-grid logs” | “Used public UCI household measurements and a separate seeded virtual building” |
| “Built from scratch alone over seven days” | “Used an AI-assisted scaffold, reproduced it and contributed documented experiments” |

External context was reviewed for this build on October 15, 2026. Historical source
dates are stated explicitly; no claim is made about live vacancies or future markets.
