# ⚡ MASTER PROMPT — AI-Powered Energy Consumption Forecasting System

> The complete, upgraded engineering specification and master prompt blueprint. Use this as your reference architecture, academic defense guide, or whenever rebuilding the system. It defines non-negotiable acceptance criteria, rigorous evaluation metrics, and verified deliverables.

---

## THE MASTER PROMPT

### ROLE
You are a senior team combining: Senior Machine Learning Engineer, Smart Grid & Energy Systems Analyst, Full-Stack Python/JavaScript Developer, and Technical Mentor for university and competitive placement portfolios.

### MISSION
Build a complete, runnable, production-grade project titled **"AI-Powered Energy Consumption Forecasting System"** that a student or engineer can showcase with full transparency:
- Forecasting hourly electricity consumption across residential and commercial buildings.
- Executing leakage-free time-series backtesting (rolling origin evaluation).
- Providing an interactive decision-support sandbox for load-shifting and peak shaving under time-of-use tariffs.
- Running on CPU-only hardware in minutes with zero external API dependencies.
- Delivering both an interactive offline web console and a RESTful Flask API.

### NON-NEGOTIABLE ACCEPTANCE CRITERIA
1. **End-to-End Execution**: `python main.py --dataset synthetic` and `python main.py --dataset uci` run without manual intervention.
2. **Strict Causal Discipline**: No data leakage. Features (lags, rolling stats, cyclic time encodings) must only use historical observations available before forecast cutoff.
3. **Multi-Model Comparison**: Evaluates naive baselines (persistence, 24h lag, 168h seasonal lag) alongside Ridge Regression, Random Forest, and Scaled Multi-Layer Perceptron (MLP).
4. **Leakage-Free Validation**: Model selection is performed exclusively on validation splits; test evaluation is reserved strictly for final reporting.
5. **Thermodynamic / Physical Honesty**: Virtual load-shifting simulation must conserve total energy (kWh). Under constant emission intensity, demand shifting shifts cost, not total energy consumed.
6. **Automated Testing & Integrity**: Minimum 60+ pytest unit/integration tests and automated artifact validation checks verifying reproducible numerical parity.

### DELIVERABLES
- **A. Problem Explanation**: Plain language and technical framing, industry applications (smart buildings, demand response, microgrids).
- **B. Architectural Blueprint**: Modular pipeline separation: Data Ingestion → Cleaning & Imputation → Feature Engineering → Model Training → Multi-Horizon Forecasting → Decision Simulation → Visualization & Dashboard.
- **C. Data Pipelines**: Deterministic synthetic building generator (17,520 hours) and public UCI Individual Household Electric Power Consumption dataset adapter with missingness handling.
- **D. Feature Engineering**: Lag features (1h, 2h, 3h, 24h, 168h), rolling window aggregates (6h, 24h mean/std), cyclical calendar encoding (hour-of-day, day-of-week sine/cosine), and temperature/occupancy indicators.
- **E. Evaluation & Metrics**: One-step and recursive multi-step (24h/168h) horizon evaluation with RMSE, MAE, MAPE, and R² metrics.
- **F. Interactive Dashboard**: Dark-themed, responsive web console featuring historical charts, multi-horizon forecasts with empirical prediction intervals, model comparison tables, and interactive load shifting simulator.
- **G. REST API**: Endpoints for dataset switching, horizon configuration, model selection, live inference, and CSV report export.
- **H. Comprehensive Verification**: Complete documentation, data cards, model cards, audit trails, and automated test suites.
